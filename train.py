import sys
import os
import math
import torch
import argparse
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from model import EfficientNet_b0
from CNN.PLAM.utils import compute_window_nums, get_coordinates, feature_location, proposal_generation

from Utils.my_dataset import MyDataset

# class UnNormalize(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std
#
#     def __call__(self, tensor):
#         for t, m, s in zip(tensor, self.mean, self.std):
#             t.mul_(s).add_(m)
#         return tensor


# 处理图像时使用的步幅。
stride = 32
input_size = 448
window_number = [3, 2, 1]  # 包含不同配置下窗口数量的整数列表
# 不同窗口的长宽比率列表
len_wid_ratio = [[6, 6], [5, 7], [7, 5],
                 [8, 8], [6, 10], [10, 6], [7, 9], [9, 7],
                 [10, 10], [9, 11], [11, 9], [8, 12], [12, 8]]
# window_number = [2, 3, 2]  # 包含不同配置下窗口数量的整数列表
# len_wid_ratio = [[4, 4], [3, 5], [5, 3],
#                  [6, 6], [5, 7], [7, 5],
#                  [8, 8], [6, 10], [10, 6], [7, 9], [9, 7], [7, 10], [10, 7]]
sum_window = sum(window_number)  # 窗口总数

# 交并比
iou_threshs = [0.5, 0.5, 0.5]
# 获取对应长宽比率的滑动窗口总数
ratios_window_sum = compute_window_nums(len_wid_ratio, stride, input_size)
# 获取3个编号的滑动窗口总数[0, 241, 139, 211]
window_nums_sum = [0, sum(ratios_window_sum[:3]), sum(ratios_window_sum[3:6]), sum(ratios_window_sum[6:])]
# 获取对应的滑动窗口的索引，类似给每个ratios中的滑动窗口进行编号list([0~81], [0~80], ...,[0~21],[0~21])
indexs = [np.arange(0, window_sum).reshape(-1, 1) for window_sum in ratios_window_sum]
# 获取了滑动窗口的坐标列表
coordinates = [get_coordinates(index, stride, input_size, len_wid_ratio[i]) for i, index in enumerate(indexs)]
# 将coordinates按行拼接 [591,4]
all_window_coordinates = np.concatenate(coordinates, 0)


# 计算loss函数
def train(model, optimizer, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
    model.train()

    raw_correct, located_correct, total, mean_raw_loss, mean_located_loss, mean_proposal_loss, mean_total_loss = \
        0, 0, 0, 0, 0, 0, 0

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):

        images, labels = data
        images, labels = images.to(device), labels.to(device)
        total += labels.size(0)

        optimizer.zero_grad()
        attention_map, ori_fc_result = model(images)
        ori_loss = loss_function(ori_fc_result, labels)

        batch_size, _, _, _ = attention_map.shape
        fc_weight = model.state_dict()['linear.weight']
        located_images = feature_location(attention_map.detach(), fc_weight, ori_fc_result, images)
        # unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # located_image = unorm(located_images).mul(255).byte()
        # located_image = np.squeeze(located_image.detach().cpu().numpy())
        # located_image = np.transpose(located_image, [1, 2, 0])
        # plt.figure()
        # plt.imshow(located_image)
        # plt.axis('off')
        # plt.show()
        located_attention_map, located_fc_result = model(located_images.clone().detach())
        located_loss = loss_function(located_fc_result, labels)

        proposalN_indices = proposal_generation(proposalN=sum_window,
                                                conv_result=located_attention_map.detach(),
                                                ratios=len_wid_ratio,
                                                window_nums_sum=window_nums_sum,
                                                N_list=window_number,
                                                iou_threshs=iou_threshs,
                                                device=device,
                                                coordinates_cat=all_window_coordinates)

        proposal_images = torch.zeros([batch_size, sum_window, 3, 224, 224]).to(device)
        for i in range(batch_size):
            for j in range(sum_window):
                # 取出第对应proposal窗口的坐标
                [x0, y0, x1, y1] = all_window_coordinates[proposalN_indices[i, j]]
                # 通过双线性插值，将AOLM_images对应上面的坐标位置的像素，复制给APPM_images相应位置，从而达到提取关信息的目的
                proposal_images[i:i + 1, j] = F.interpolate(located_images[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)],
                                                            size=(224, 224), mode='bilinear', align_corners=True)

        proposal_images = proposal_images.reshape(batch_size * sum_window, 3, 224, 224)  # [N*4, 3, 224, 224]

        _, proposal_fc_result = model(proposal_images.detach())  # [N*4, 2048]
        proposal_loss = loss_function(proposal_fc_result, labels.unsqueeze(1).repeat(1, sum_window).view(-1))
        total_loss = ori_loss + located_loss + proposal_loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        mean_raw_loss = (mean_raw_loss * step + ori_loss.detach()) / (step + 1)
        mean_located_loss = (mean_located_loss * step + located_loss.detach()) / (step + 1)
        mean_proposal_loss = (mean_proposal_loss * step + proposal_loss.detach()) / (step + 1)
        mean_total_loss = (mean_total_loss * step + total_loss.detach()) / (step + 1)

        _, raw_pred = torch.max(ori_fc_result.detach(), 1)
        raw_correct += raw_pred.eq(labels.detach()).cpu().sum()

        _, located_pred = torch.max(located_fc_result.detach(), 1)
        located_correct += located_pred.eq(labels.detach()).cpu().sum()

        data_loader.desc = "[epoch {}]: raw mean loss: {:.3f}, located mean loss: {:.3f}, proposal mean loss: {:.3f}, " \
                           "total mean loss: {:.3f}, raw acc: {:.3f}, located acc: {:.3f}".format(epoch,
                                                                                                  mean_raw_loss.item(),
                                                                                                  mean_located_loss.item(),
                                                                                                  mean_proposal_loss.item(),
                                                                                                  mean_total_loss.item(),
                                                                                                  raw_correct / total,
                                                                                                  located_correct / total)

    return mean_raw_loss.item(), mean_located_loss.item(), mean_proposal_loss.item(), mean_total_loss.item(), \
           raw_correct / total, located_correct / total


# 计算准确度

def evaluate(model, test_loader, device, epoch):
    model.eval()
    raw_correct, located_correct, total = 0, 0, 0

    with torch.no_grad():
        data_loader = tqdm(test_loader, file=sys.stdout)
        for i, data in enumerate(data_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            total += labels.size(0)

            attention_map, ori_fc_result = model(images)

            fc_weight = model.state_dict()['linear.weight']
            located_images = feature_location(attention_map.detach(), fc_weight, ori_fc_result, images)
            located_images = located_images.clone().detach()
            _, located_fc_result = model(located_images)

            _, raw_pred = torch.max(ori_fc_result.detach(), 1)
            raw_correct += raw_pred.eq(labels.detach()).cpu().sum()
            # AOLM
            _, located_pred = torch.max(located_fc_result.detach(), 1)
            located_correct += located_pred.eq(labels.detach()).cpu().sum()

            data_loader.desc = "[epoch {}]: test raw acc: {:.3f}, test located acc: {:.3f}".\
                format(epoch, raw_correct / total, located_correct / total)

    return raw_correct / total, located_correct / total


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(args)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(448),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize(448),
                                    transforms.CenterCrop(448),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataset(root=os.path.join(args.data_path, "train.txt"),
                              transforms=data_transform["train"])

    # 实例化验证数据集
    test_dataset = MyDataset(root=os.path.join(args.data_path, "test.txt"),
                             transforms=data_transform["test"])

    print("using {} images for training, {} images for testing.".format(len(train_dataset), len(test_dataset)))

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw)

    model = EfficientNet_b0(num_classes=args.num_classes).to(device)

    # 如果存在预训练权重则载入
    if args.weights != "":
        if os.path.exists(args.weights):
            model_weghts_path = args.weights
            weights_dict = torch.load(model_weghts_path)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if ('se' not in k) and ('classifier' not in k)}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    temp_acc = []
    for epoch in range(args.epochs):
        # train
        raw_loss, located_loss, proposal_loss, total_loss, raw_acc, located_acc = train(model=model,
                                                                                        optimizer=optimizer,
                                                                                        data_loader=train_loader,
                                                                                        device=device,
                                                                                        epoch=epoch)
        scheduler.step()

        with open('temp/PLAM_train.txt', 'a') as f:
            f.write('epoch : %d, raw_loss : %.3f, located_loss : %.3f, proposal_loss : %.3f, total_loss : %.3f, '
                    'raw_acc : %.3f, located_acc : %.3f' % (epoch, raw_loss, located_loss, proposal_loss, total_loss,
                                                            raw_acc, located_acc) + '\n')

        # validate
        raw_accuracy, located_accuracy = evaluate(model=model, test_loader=test_loader, device=device, epoch=epoch)

        with open('temp/PLAM_test.txt', 'a') as w:
            w.write('epoch : %d, raw_accuracy : %.3f, located_accuracy : %.3f' %
                    (epoch, raw_accuracy, located_accuracy) + '\n')

        if epoch == 0:
            temp_acc.append(located_accuracy)
            torch.save(model.state_dict(), "temp/PLAM_weights.pth")
        else:
            temp_acc.append(located_accuracy)
            if temp_acc[1] > temp_acc[0]:
                torch.save(model.state_dict(), "temp/PLAM_weights.pth")
                del temp_acc[0]
            else:
                del temp_acc[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=102)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--data-path', type=str, default="")
    opt = parser.parse_args()

    main(opt)
