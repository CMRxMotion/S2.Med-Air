# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from monai.networks.blocks import Convolution
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys, SwinTransformerSys_en, SwinTransformerSys_de
from .swin_transformer_unet_skip_expand_decoder_sys_3d import SwinTransformerSys as SwinTransformerSys3d
logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=img_size,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=in_chans,
                                num_classes=32,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        self.conv1 = Convolution(2, 32, 16)
        self.conv2 = nn.Conv2d(16, self.num_classes, kernel_size=1,bias=False)


    def forward(self, x):
        logits = self.swin_unet(x)
        logits = self.conv1(logits)
        logits = self.conv2(logits)
        return logits

    def load_from(self, pretrained_path):
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")

    def load_from_en(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            for key in ["layers.0.blocks.1.attn_mask", "layers.1.blocks.1.attn_mask", "layers.2.blocks.1.attn_mask", "layers.3.blocks.1.attn_mask"]:
                pretrained_dict[key] = self.swin_unet_encoder.state_dict()[key]
            self.swin_unet_encoder.load_state_dict(pretrained_dict)
        else:
            print("none pretrain")


class SwinUnet3D(nn.Module):
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False):
        super(SwinUnet3D, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys3d()

    def forward(self, x):
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
def crop_or_pad(img, center_list, output_size, pad = 0):
    #img: b*h:w
    B, C, H, W = img.shape
    img_crop_list = []
    for i in range(B):
        center = center_list[i]
        left_pad = 0
        right_pad = 0
        up_pad = 0
        down_pad = 0
        if center[0] < output_size//2:
            left_pad = output_size - center[0]
        if center[0] + output_size//2 > H:
            right_pad = center[0] + output_size//2 - H
        if center[1] < output_size//2:
            up_pad = output_size - center[1]
        if center[1] + output_size//2 > W:
            down_pad = center[1] + output_size//2 - W
        img_pad = torch.nn.functional.pad(img[[i]],(up_pad, down_pad, left_pad, right_pad), value = pad)
        center_x = center[0] + left_pad
        center_y = center[1] + up_pad
        img_crop = img_pad[:, :, (center_x-output_size//2):(center_x+output_size//2),
                   (center_y-output_size//2):(center_y+output_size//2)]
        img_crop_list.append(img_crop)
    return torch.cat(img_crop_list, dim=0)



class SwinUnet_s2(nn.Module):
    def __init__(self, config):
        super(SwinUnet_s2, self).__init__()
        self.swin_3d = SwinUnet3D(config, img_size=448, num_classes=2)
        self.swin_2d = SwinUnet(config, img_size=224, num_classes=4, in_chans=3)
        coords_h = torch.arange(448)
        coords_w = torch.arange(448)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            self.coords = torch.stack(torch.meshgrid([coords_h, coords_w])).permute(1, 2, 0).cuda()
        else:
            self.coords = torch.stack(torch.meshgrid([coords_h, coords_w])).permute(1, 2, 0)

    def forward(self, x):
        B = x.shape[0]
        x_3d = x.permute(1,2,3,0)
        x_3d = x_3d.unsqueeze(0)
        if B < 12:
            x_3d = torch.nn.functional.pad(x_3d, [0,12-B])
        else:
            x_3d = x_3d[:,:,:,:,:12]
        logits = self.swin_3d(x_3d) # 1*1*448*448*12
        if B > 12:
            logits = torch.nn.functional.pad(logits, [0,B-12])
        else:
            logits = logits[:,:,:,:,:B]
        logits = logits.squeeze(0).permute(3,0,1,2) #D*2*448*448

        logits_detach = torch.nn.functional.softmax(logits, dim=1)[:,1].detach() #D*448*448
        new_img = torch.cat([x,logits], dim=1)
        center_x = torch.sum(logits_detach * self.coords[:,:,0], dim=[1,2]) / torch.sum(logits_detach, dim=[1,2])
        center_y = torch.sum(logits_detach * self.coords[:, :, 1], dim=[1, 2]) / torch.sum(logits_detach, dim=[1, 2])
        center_list = torch.cat([center_x.unsqueeze(1), center_y.unsqueeze(1)], dim=1).type(torch.int)
        new_img = crop_or_pad(new_img, center_list, 224)
        logits_refine = self.swin_2d(new_img)
        center_list = 224-center_list + 112
        logits_refine_back = logits_refine[:,[0]]
        logits_refine_other = logits_refine[:,1:]
        logits_refine_back = crop_or_pad(logits_refine_back, center_list, 448, 100)
        logits_refine_other = crop_or_pad(logits_refine_other, center_list, 448)
        logits_refine = torch.cat([logits_refine_back, logits_refine_other], dim=1)
        return logits, logits_refine





