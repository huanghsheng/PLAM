import sys
import os
import torch
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from model import EfficientNet_b0
from CNN.my_dataset import MyDataset
from Utils import compute_window_nums, get_coordinates, feature_location, proposal_generation


def evaluate(model, test_loader, device):
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            attention_map, ori_fc_result = model(images)

            fc_weight = model.state_dict()['linear.weight']
            located_images = feature_location(attention_map.detach(), fc_weight, ori_fc_result, images)
            located_images = located_images.clone().detach()
            _, Located_fc_result = model(located_images)

    return Located_fc_result


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(args)

    data_transform = {
        "test": transforms.Compose([transforms.Resize(448),
                                    transforms.CenterCrop(448),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化验证数据集
    test_dataset = MyDataset(root=os.path.join(args.data_path, "test.txt"),
                             transforms=data_transform["test"])

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers

    print("using {} images for testing.".format(len(test_dataset)))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw)

    model = EfficientNet_b0(num_classes=args.num_classes).to(device)

    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    start_time = time.time()
    with torch.no_grad():
        # data_loader = tqdm(test_loader, file=sys.stdout)
        # for step, data in enumerate(data_loader):
        #     images, labels = data
        #     images, labels = images.cuda(), labels.cuda()
        #     attention_map, ori_fc_result = model(images)
        evaluate(model, test_loader, device)
    end_time = time.time()

    average_inference_time = (end_time - start_time) / len(test_dataset)
    inference_speed = 1 / average_inference_time  # 推理速度
    print(f"Average Inference Time: {round((average_inference_time * 1000), 2)} ms")
    print(f"Processing per second: {round(inference_speed, 1)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=102)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--weights', type=str, default='', help='initial weights pat h')
    parser.add_argument('--data-path', type=str, default="")
    opt = parser.parse_args()
    # Average Inference Time: 2.51 ms
    # +evaluate Average Inference Time: 4.97 ms


    main(opt)
