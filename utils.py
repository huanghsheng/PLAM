import numpy as np
import torch
import torchvision


# 计算不同ratio下其滑动窗口数
def compute_window_nums(len_wid_ratio, stride, input_size):
    size = input_size / stride
    window_nums = []
    for _, ratio in enumerate(len_wid_ratio):
        window_nums.append(int((size - ratio[0]) + 1) * int((size - ratio[1]) + 1))
    return window_nums


# 返回滑动窗口的坐标列表。index：滑动窗口索引
def get_coordinates(index, stride, image_size, len_wid_ratio):
    batch, _ = index.shape
    size = int(image_size / stride)
    coordinates = []
    # 这里对应的是feature map中每列的窗口数量
    column_window_num = (size - len_wid_ratio[1]) + 1

    # 获取了ratios[i]对应num的滑动窗口坐标
    for j, indice in enumerate(index):
        # 计算滑动窗口在x、y方向的起始位置
        x_indice = indice // column_window_num
        y_indice = indice % column_window_num
        # 计算滑动窗口的坐标
        x_lefttop = x_indice * stride - 1
        y_lefttop = y_indice * stride - 1
        x_rightlow = x_lefttop + len_wid_ratio[0] * stride
        y_rightlow = y_lefttop + len_wid_ratio[1] * stride
        # for image
        x_lefttop = max(0, x_lefttop)
        y_lefttop = max(0, y_lefttop)

        coordinates.append(np.array((x_lefttop, y_lefttop, x_rightlow, y_rightlow), dtype=object).reshape(1, 4))

    coordinates = np.array(coordinates).reshape(batch, 4).astype(int)

    return coordinates


def feature_location(Attention_map, fc_weight, ori_fc_result, image):
    batch, channel, H, W = Attention_map.size()
    images = image.clone()
    _, _, imgH, imgW = images.size()
    # channel维度做平均
    located_map = torch.zeros_like(Attention_map.mean(1))
    theta = 0.5
    padding_ratio = 0.1

    for batch_index in range(batch):
        # 获取每张照片对应的激活图
        map_tpm = Attention_map[batch_index]
        # [C, H * W]
        map_tpm = map_tpm.reshape(channel, H * W)
        # [H * W, C]
        map_tpm = map_tpm.permute([1, 0])

        # 对应专家对该张图片的分类
        pred_tmp = ori_fc_result[batch_index]
        pred_ind = pred_tmp.argmax()
        p2_tmp = fc_weight[pred_ind].unsqueeze(1)
        map_tpm = torch.mm(map_tpm, p2_tmp)
        # 得到最终的feature map [14,14]
        located_map[batch_index] = map_tpm.reshape(H, W)

    attention_map = located_map.clone().detach()

    for batch_index in range(batch):
        # 获取每张照片
        image_tmp = images[batch_index]
        # 获取其对应的feature map,并升维度到[1, 1, 14, 14]
        upsample_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
        # 通过插值操作将[1, 1, 14, 14] -> [448,448]
        upsample_tpm = torch.nn.functional.interpolate(upsample_tpm, size=(imgH, imgW), mode='bilinear',
                                                       align_corners=True).squeeze()
        # 归一化，并将大于theta的值置为ture
        upsample_tpm = (upsample_tpm - upsample_tpm.min()) / (upsample_tpm.max() - upsample_tpm.min() + 1e-6)
        upsample_tpm = upsample_tpm >= theta
        # 以张量的形式返回非零元素的位置索引（此处将False当成了0）
        nonzero_indices = torch.nonzero(upsample_tpm, as_tuple=False)
        # 获取最小与最大的y坐标
        height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
        height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
        # 获取最小与最大的x坐标
        width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
        width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
        # 此处获取了feature map中，信息最丰富的区域
        image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)
        # 再将该图像处理成原图像的大小
        image_tmp = torch.nn.functional.interpolate(image_tmp, size=(imgH, imgW), mode='bilinear',
                                                    align_corners=True).squeeze()

        images[batch_index] = image_tmp

    return images


def proposal_generation(proposalN, conv_result, ratios, window_nums_sum, N_list, iou_threshs, device, coordinates_cat):
    batch, channels, _, _ = conv_result.size()

    avgpools = [torch.nn.MaxPool2d(ratios[i], 1) for i in range(len(ratios))]
    avgs = [avgpools[i](conv_result) for i in range(len(ratios))]

    # 得到分数
    fm_sum = [torch.sum(avgs[i], dim=1) for i in range(len(ratios))]
    all_scores = torch.cat([fm_sum[i].view(batch, -1, 1) for i in range(len(ratios))], dim=1)
    windows_scores_np = all_scores.data.cpu().numpy()

    proposalN_indices = []
    for i, scores in enumerate(windows_scores_np):
        indices_results = []
        # 此处获取每个编号对应的滑动窗口索引
        for j in range(len(window_nums_sum) - 1):
            indices_results.append(
                # 此处获取的是信息最丰富的滑动窗口的索引
                # 对应scores中按[0, 241,139,211]顺序提取对应数量的滑动窗口的分数
                cluster_nms(scores[sum(window_nums_sum[:j + 1]):sum(window_nums_sum[:j + 2])],
                            proposalN=N_list[j],
                            iou_threshs=iou_threshs[j],
                            coordinates=coordinates_cat[sum(window_nums_sum[:j + 1]):sum(window_nums_sum[:j + 2])])
                + sum(window_nums_sum[:j + 1]))
        proposalN_indices.append(np.concatenate(indices_results, 1))

    proposalN_indices = torch.from_numpy(np.array(proposalN_indices).reshape(batch, proposalN)).to(device)

    # 获取batch张照片所选中的滑动窗口它们各自的分数
    # proposalN_windows_scores = torch.cat(
    #     [torch.index_select(all_score, dim=0, index=proposalN_indices[i]) for i, all_score in
    #      enumerate(all_scores)], 0).reshape(
    #     batch, proposalN)

    return proposalN_indices


def cluster_nms(scores_np, proposalN, iou_threshs, coordinates):
    if not (type(scores_np).__module__ == 'numpy' and len(scores_np.shape) == 2 and scores_np.shape[1] == 1):
        raise TypeError('score_np is not right')

    windows_num = scores_np.shape[0]
    # 将分数与框进行拼接
    sores_coordinates = np.concatenate((scores_np, coordinates), 1)
    # 获取了每个分数其对应的索引
    indices = np.argsort(sores_coordinates[:, 0])
    # [score, box, index]
    indices_coordinates = np.concatenate((sores_coordinates, np.arange(0, windows_num).reshape(windows_num, 1)), 1)[
        indices]

    # 获取分数
    scores = torch.from_numpy(indices_coordinates[:, 0]).unsqueeze(1)
    # 获取边界框
    boxes = torch.from_numpy(indices_coordinates[:, 1:5]).float()
    # 将分数从高到低进行排序
    scores, idx = scores.sort(0, descending=True)
    # 将边界框从高到低进行排序
    boxes = boxes[idx].squeeze(1)
    # IoU矩阵，上三角化
    iou = torchvision.ops.box_iou(boxes, boxes).triu_(diagonal=1)
    C = iou
    for i in range(iou.size(0)):
        A = C
        maxA = A.max(dim=0)[0]  # 列最大值向量
        E = (maxA < iou_threshs).float().unsqueeze(1).expand_as(A)  # 对角矩阵E的替代
        C = iou.mul(E)  # 按元素相乘
        if A.equal(C):  # 终止条件
            break
    keep = (maxA < iou_threshs).float()  # 列最大值向量，二值化

    # 获取符合条件的提议框对应的分数
    selected_scores = scores[keep.bool()].numpy()

    # 获取 indices_coordinates 中的分数列
    all_scores = indices_coordinates[:, 0]

    # 寻找选中框对应的索引
    indices_results = []
    for score in selected_scores:
        index = np.where(all_scores == score)[0][0]
        indices_results.append(index)
        if len(indices_results) == proposalN:
            return np.array(indices_results).reshape(1, proposalN).astype(np.int)

    return np.array(indices_results).reshape(1, -1).astype(np.int)
