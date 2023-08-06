import torch
import numpy as np
from lib.utils.misc import NestedTensor
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import cv2
import os

class Preprocessor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return NestedTensor(img_tensor_norm, amask_tensor)

class Preprocessor_wo_mask(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        return img_tensor_norm


class PreprocessorX(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return img_tensor_norm, amask_tensor


class PreprocessorX_onnx(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        """img_arr: (H,W,3), amask_arr: (H,W)"""
        # Deal with the image patch
        img_arr_4d = img_arr[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        img_arr_4d = (img_arr_4d / 255.0 - self.mean) / self.std  # (1, 3, H, W)
        # Deal with the attention mask
        amask_arr_3d = amask_arr[np.newaxis, :, :]  # (1,H,W)
        return img_arr_4d.astype(np.float32), amask_arr_3d.astype(np.bool)


def vis_feature_maps(features, resolution, img, save_path='.'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shape_f = [resolution, resolution]
    w, h, c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    featuretype = "t" if resolution == 8 else "s"

    features_mean = []
    for feature in features:
        if resolution == 8:
            feature = feature[:, :64, :]
        elif resolution == 16:
            feature = feature[:, 64:, :]
        features_mean.append(feature.mean(dim=2).squeeze().reshape(shape_f).cpu().numpy())

    block_num = 0
    for feature in features_mean:
        feature = np.maximum(feature, 0)
        feature /= np.max(feature)
        # feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
        feature = cv2.resize(feature, (w, h))
        feature = np.uint8(255 * feature)
        feature = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
        figure = cv2.addWeighted(img, 1, feature, 0.5, 0)
        cv2.imwrite(save_path + '/' + featuretype + '_Block{}_feature.png'.format(block_num), figure)
        block_num += 1

    del features_mean


def vis_attn_maps(attn_weights, q_w, k_w, skip_len1, skip_len2, x1, x2, x1_title, x2_title, save_path='.', idxs=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shape1 = [q_w, q_w]
    shape2 = [k_w, k_w]

    attn_weights_mean = []
    for attn in attn_weights:
        attn_weights_mean.append(attn[..., skip_len1:(skip_len1+q_w**2), skip_len2:(skip_len2+k_w**2)].mean(dim=1).squeeze().reshape(shape1+shape2).cpu())

    # downsampling factor
    fact = 16

    if idxs is None:
        idxs = [(64, 64)]

    block_num = 0
    idx_o = idxs[0]
    for attn_weight in attn_weights_mean:
        fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
        ax = fig.add_subplot(111)
        idx = (idx_o[0] // fact, idx_o[1] // fact)
        ax.imshow(attn_weight[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
        ax.axis('off')
        # ax.set_title(f'Stage2-Block{block_num}')
        plt.savefig(save_path + '/Block{}_attn_weight.png'.format(block_num))
        plt.close()
        block_num += 1

    fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    x2_ax = fig.add_subplot(111)
    x2_ax.imshow(x2)
    x2_ax.axis('off')
    plt.savefig(save_path + '/{}.png'.format(x2_title))
    plt.close()

    # the reference points as red circles
    fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    x1_ax = fig.add_subplot(111)
    x1_ax.imshow(x1)
    for (y, x) in idxs:
        # scale = im.height / img.shape[-2]
        x = ((x // fact) + 0.5) * fact
        y = ((y // fact) + 0.5) * fact
        x1_ax.add_patch(plt.Circle((x, y), fact // 2, color='r'))
        # x1_ax.set_title(x1_title)
        x1_ax.axis('off')
    plt.savefig(save_path+'/{}.png'.format(x1_title))
    plt.close()

    del attn_weights_mean
