import argparse
from copy import deepcopy
from typing import Tuple, Union, List

import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax, save_segmentation_nifti
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Process, Queue
import torch
import SimpleITK as sitk
import shutil
from multiprocessing import Pool
from nnunet.postprocessing.connected_components import load_remove_save, load_postprocessing
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.one_hot_encoding import to_one_hot
from config import get_config
import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from shutil import copyfile
from torchvision import transforms
from torch.nn import functional as F
import nibabel as nib
from monai.networks.nets import SwinUNETR
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose as Compose_nn
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform, \
    MaskTransform, ConvertSegmentationToRegionsTransform
from nnunet.training.data_augmentation.pyramid_augmentations import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
try:
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
except ImportError as ie:
    NonDetMultiThreadedAugmenter = None

from networks.vision_transformer_original import SwinUnet


from monai.transforms import (
    ScaleIntensityRangePercentiles,
    NormalizeIntensity,
    Compose
)
import monai

import ttach as tta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    args = parser.parse_args()
    args.cfg = "./configs/swin_tiny_patch4_window7_224_lite.yaml"
    args.opts = None

    # merge from specific arguments
    args.batch_size = 1
    args.zip = False
    args.cache_mode = 'part'
    args.resume = False
    args.accumulation_steps = False
    args.use_checkpoint = False
    args.amp_opt_level = '01'
    args.tag = None
    args.eval = True
    args.throughput = False
    config = get_config(args)

    nnUnet2_predict(args)
    label_dir = './nnunet2dtmp/'
    save_dir = './unetr/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    models = swin_unetr_loader()
    swin_data_trans = swin_transformers()
    trans = Compose([ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_min=0., b_max=1.),
                     NormalizeIntensity()
                     ])

    keep = monai.transforms.KeepLargestConnectedComponent()
    keep1 = monai.transforms.KeepLargestConnectedComponent(independent = False)
    files = sorted(os.listdir(args.input))
    labels = sorted(os.listdir(label_dir))
    tta_models = []
    for model in models:
        model.eval()
        model_softmax = lambda x: F.softmax(model(x), dim = 1)
        tta_model = tta.SegmentationTTAWrapper(model_softmax, tta.Compose([tta.transforms.Rotate90([0, 90, 180, 270])]),
                                               merge_mode='mean')
        tta_models.append(tta_model)
    with torch.no_grad():
        for img, lab in zip(files, labels):
            img_file = nib.load(os.path.join(args.input, img))
            label_file = nib.load(os.path.join(label_dir, lab))
            images = img_file.get_fdata()
            sudo_label = label_file.get_fdata()
            pred_label = sudo_label.copy()
            tmp = np.where(sudo_label > 0)
            x_min = np.min(tmp[0])
            x_max = np.max(tmp[0])
            x_mid = (x_min + x_max) // 2
            y_min = np.min(tmp[1])
            y_max = np.max(tmp[1])
            y_mid = (y_min + y_max) // 2
            images_crop = images[(x_mid - 112):(x_mid + 112), (y_mid - 112):(y_mid + 112), :]
            labels_crop = sudo_label[(x_mid - 112):(x_mid + 112), (y_mid - 112):(y_mid + 112), :]
            d = images_crop.shape[2]
            for i in range(d):
                images_slice = trans(torch.tensor(images_crop[:, :, i, 0]).unsqueeze(0).unsqueeze(0).float())
                labels_slice = np.expand_dims(labels_crop[:, :, i], 0)
                d = {'data': images_slice, 'seg': labels_slice}
                out = swin_data_trans(**d)
                images_batch = out['data']
                images_slice = (images_batch - torch.tensor(
                    images_batch.cpu().numpy().min(axis=(1, 2, 3)).reshape([-1, 1, 1, 1]))) \
                               / torch.tensor(images_batch.cpu().numpy().max(axis=(1, 2, 3)).reshape([-1, 1, 1, 1]) - \
                                              images_batch.cpu().numpy().min(axis=(1, 2, 3)).reshape(
                                                  [-1, 1, 1, 1]))
                images_slice = images_slice.cuda()
                pred = tta_models[0](images_slice)
                for tta_model in tta_models[1:]:
                    pred += tta_model(images_slice)
                pred = torch.argmax(pred, dim=1)
                pred = keep1(pred)
                pred = keep(pred)[0]
                if (torch.sum((pred == 1) | (pred == 2)) == 0):
                    pred[pred == 2] = 0
                pred_label[(x_mid - 112):(x_mid + 112), (y_mid - 112):(y_mid + 112), i] = pred.cpu().numpy()
            pred_label = keep(np.expand_dims(pred_label, 0))[0].astype(int)
            nib.Nifti1Image(pred_label, label_file.affine).to_filename(save_dir + lab)
    for model in models:
        del model
    torch.cuda.empty_cache()
    models = swin_unet_loader(config)
    for model in models:
        model.eval()
        model_softmax = lambda x: F.softmax(model(x), dim = 1)
        tta_model = tta.SegmentationTTAWrapper(model_softmax, tta.Compose([tta.transforms.Rotate90([0, 90, 180, 270])]),
                                               merge_mode='mean')
        tta_models.append(tta_model)
    save_dir = './unet/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    with torch.no_grad():
        for img, lab in zip(files, labels):
            img_file = nib.load(os.path.join(args.input, img))
            label_file = nib.load(os.path.join(label_dir, lab))
            images = img_file.get_fdata()
            sudo_label = label_file.get_fdata()
            pred_label = sudo_label.copy()
            tmp = np.where(sudo_label > 0)
            x_min = np.min(tmp[0])
            x_max = np.max(tmp[0])
            x_mid = (x_min + x_max) // 2
            y_min = np.min(tmp[1])
            y_max = np.max(tmp[1])
            y_mid = (y_min + y_max) // 2
            images_crop = images[(x_mid - 112):(x_mid + 112), (y_mid - 112):(y_mid + 112), :]
            labels_crop = sudo_label[(x_mid - 112):(x_mid + 112), (y_mid - 112):(y_mid + 112), :]
            d = images_crop.shape[2]
            for i in range(d):
                images_slice = trans(torch.tensor(images_crop[:, :, i, 0]).unsqueeze(0).unsqueeze(0).float())
                labels_slice = np.expand_dims(labels_crop[:, :, i], 0)
                d = {'data': images_slice, 'seg': labels_slice}
                out = swin_data_trans(**d)
                images_batch = out['data']
                images_slice = (images_batch - torch.tensor(
                    images_batch.cpu().numpy().min(axis=(1, 2, 3)).reshape([-1, 1, 1, 1]))) \
                               / torch.tensor(images_batch.cpu().numpy().max(axis=(1, 2, 3)).reshape([-1, 1, 1, 1]) - \
                                              images_batch.cpu().numpy().min(axis=(1, 2, 3)).reshape(
                                                  [-1, 1, 1, 1]))
                images_slice = images_slice.cuda()
                pred = tta_models[0](images_slice)
                for tta_model in tta_models[1:]:
                    pred += tta_model(images_slice)
                pred = torch.argmax(pred, dim=1)
                pred = keep1(pred)
                pred = keep(pred)[0]
                if (torch.sum((pred == 1) | (pred == 2)) == 0):
                    pred[pred == 2] = 0
                pred_label[(x_mid - 112):(x_mid + 112), (y_mid - 112):(y_mid + 112), i] = pred.cpu().numpy()
            pred_label = keep(np.expand_dims(pred_label, 0))[0].astype(int)
            nib.Nifti1Image(pred_label, label_file.affine).to_filename(save_dir + lab)
    for model in models:
        del model
    torch.cuda.empty_cache()
    nn_2d = './nnunet2dtmp/'
    unetr = './unetr/'
    unet = './unet/'
    nn_2d_file = sorted(os.listdir(nn_2d))
    unetr_file = sorted(os.listdir(unetr))
    unetr_file = sorted([i for i in unetr_file if i.split('.')[-1] == 'gz'])
    unet_file = sorted(os.listdir(unet))
    unet_file = sorted([i for i in unet_file if i.split('.')[-1] == 'gz'])
    save_dir = args.output
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for unetr_,nn2d_,unet_ in zip(unetr_file, nn_2d_file, unet_file):
        affine = nib.load(unetr + unetr_).affine
        file2 = F.one_hot(torch.tensor(nib.load(nn_2d + nn2d_).get_fdata()).long())
        file3 = F.one_hot(torch.tensor(nib.load(unet + unet_).get_fdata()).long())
        file4 = F.one_hot(torch.tensor(nib.load(unetr + unetr_).get_fdata()).long())
        file = (file2 + file3 + file4)
        file = torch.argmax(file, dim=3)
        nib.Nifti1Image(file.numpy(), affine).to_filename(os.path.join(save_dir, unetr_))
    print("finished")
    return

def swin_unet_loader(config):
    model1 = SwinUnet(config, img_size=224, num_classes=4, in_chans=1).cuda()
    model2 = deepcopy(model1)
    model3 = deepcopy(model1)
    model4 = deepcopy(model1)
    model5 = deepcopy(model1)
    checkpoint1 = torch.load('./model_weights/unet/fold_0/best.pth.tar')
    model1.load_state_dict(checkpoint1['state_dict'])
    checkpoint2 = torch.load('./model_weights/unet/fold_1/best.pth.tar')
    model2.load_state_dict(checkpoint2['state_dict'])
    checkpoint3 = torch.load('./model_weights/unet/fold_2/best.pth.tar')
    model3.load_state_dict(checkpoint3['state_dict'])
    checkpoint4 = torch.load('./model_weights/unet/fold_3/best.pth.tar')
    model4.load_state_dict(checkpoint4['state_dict'])
    checkpoint5 = torch.load('./model_weights/unet/fold_4/best.pth.tar')
    model5.load_state_dict(checkpoint5['state_dict'])
    return model1, model2, model3, model3, model4

def swin_unetr_loader():
    model1 = SwinUNETR(img_size=224,
                      in_channels=1,
                      out_channels=4,
                      feature_size=48,
                      drop_rate=0,
                      attn_drop_rate=0,
                      dropout_path_rate=0.1,
                      use_checkpoint=False,
                      spatial_dims=2
                      ).cuda()
    model2 = deepcopy(model1)
    model3 = deepcopy(model1)
    model4 = deepcopy(model1)
    model5 = deepcopy(model1)
    checkpoint1 = torch.load('./model_weights/unetr/fold_0/best.pth.tar')
    model1.load_state_dict(checkpoint1['state_dict'])
    checkpoint2 = torch.load('./model_weights/unetr/fold_1/best.pth.tar')
    model2.load_state_dict(checkpoint2['state_dict'])
    checkpoint3 = torch.load('./model_weights/unetr/fold_2/best.pth.tar')
    model3.load_state_dict(checkpoint3['state_dict'])
    checkpoint4 = torch.load('./model_weights/unetr/fold_3/best.pth.tar')
    model4.load_state_dict(checkpoint4['state_dict'])
    checkpoint5 = torch.load('./model_weights/unetr/fold_4/best.pth.tar')
    model5.load_state_dict(checkpoint5['state_dict'])
    return model1, model2, model3, model4, model5

def swin_transformers():
    regions = None
    params = default_2D_augmentation_params
    params['selected_seg_channels'] = [0]
    params['patch_size_for_spatialtransform'] = (224, 224)
    params["scale_range"] = (0.7, 1.4)
    params["do_elastic"] = False
    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))
    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
    val_transforms.append(RenameTransform('seg', 'target', True))
    if regions is not None:
        val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))
    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose_nn(val_transforms)
    return val_transforms

def check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities):
    print("This model expects %d input modalities for each image" % expected_num_modalities)
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    maybe_case_ids = np.unique([i[:-12] for i in files])

    remaining = deepcopy(files)
    missing = []

    assert len(files) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

    # now check if all required files are present and that no unexpected files are remaining
    for c in maybe_case_ids:
        for n in range(expected_num_modalities):
            expected_output_file = c + "_%04.0d.nii.gz" % n
            if not isfile(join(input_folder, expected_output_file)):
                missing.append(expected_output_file)
            else:
                remaining.remove(expected_output_file)

    print("Found %d unique case ids, here are some examples:" % len(maybe_case_ids),
          np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)))
    print("If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc")

    if len(remaining) > 0:
        print("found %d unexpected remaining files in the folder. Here are some examples:" % len(remaining),
              np.random.choice(remaining, min(len(remaining), 10)))

    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")

    return maybe_case_ids

def preprocess_save_to_queue(preprocess_fn, q, list_of_lists, output_files, segs_from_prev_stage, classes,
                             transpose_forward):
    # suppress output
    # sys.stdout = open(os.devnull, 'w')

    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            output_file = output_files[i]
            print("preprocessing", output_file)
            d, _, dct = preprocess_fn(l)
            # print(output_file, dct)
            if segs_from_prev_stage[i] is not None:
                assert isfile(segs_from_prev_stage[i]) and segs_from_prev_stage[i].endswith(
                    ".nii.gz"), "segs_from_prev_stage" \
                                " must point to a " \
                                "segmentation file"
                seg_prev = sitk.GetArrayFromImage(sitk.ReadImage(segs_from_prev_stage[i]))
                # check to see if shapes match
                img = sitk.GetArrayFromImage(sitk.ReadImage(l[0]))
                assert all([i == j for i, j in zip(seg_prev.shape, img.shape)]), "image and segmentation from previous " \
                                                                                 "stage don't have the same pixel array " \
                                                                                 "shape! image: %s, seg_prev: %s" % \
                                                                                 (l[0], segs_from_prev_stage[i])
                seg_prev = seg_prev.transpose(transpose_forward)
                seg_reshaped = resize_segmentation(seg_prev, d.shape[1:], order=1)
                seg_reshaped = to_one_hot(seg_reshaped, classes)
                d = np.vstack((d, seg_reshaped)).astype(np.float32)
            """There is a problem with python process communication that prevents us from communicating objects 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically"""
            print(d.shape)
            if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                np.save(output_file[:-7] + ".npy", d)
                d = output_file[:-7] + ".npy"
            q.put((output_file, (d, dct)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print("error in", l)
            print(e)
    q.put("end")
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    else:
        print("This worker has ended successfully, no errors to report")
    # restore output
    # sys.stdout = sys.__stdout__

def preprocess_multithreaded(trainer, list_of_lists, output_files, num_processes=2, segs_from_prev_stage=None):
    if segs_from_prev_stage is None:
        segs_from_prev_stage = [None] * len(list_of_lists)

    num_processes = min(len(list_of_lists), num_processes)

    classes = list(range(1, trainer.num_classes))
    assert isinstance(trainer, nnUNetTrainer)
    q = Queue(1)
    processes = []
    for i in range(num_processes):
        pr = Process(target=preprocess_save_to_queue, args=(trainer.preprocess_patient, q,
                                                            list_of_lists[i::num_processes],
                                                            output_files[i::num_processes],
                                                            segs_from_prev_stage[i::num_processes],
                                                            classes, trainer.plans['transpose_forward']))
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == "end":
                end_ctr += 1
                continue
            else:
                yield item

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()  # this should not happen but better safe than sorry right
            p.join()

        q.close()

def nnUnet2_predict(args):
    for file in os.listdir(args.input):
        newfile_name = file.split('.')[0]+'_0000.nii.gz'
        copyfile(os.path.join(args.input, file), os.path.join('./nnunetinput/', newfile_name))
    input_folder = './nnunetinput/'
    output_folder = './nnunet2dtmp/'
    model = './model_weights/nnUnet_2d/'
    folds = [0, 1, 2, 3, 4]
    num_threads_preprocessing = 6
    num_threads_nifti_save = 2
    pool = Pool(num_threads_nifti_save)
    results = []
    step_size = 0.5
    all_in_gpu = None
    part_id = 0
    num_parts = 1
    expected_num_modalities = load_pickle(join(model, "plans.pkl"))['num_modalities']
    case_ids = check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in case_ids]
    trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=True,
                                                      checkpoint_name="model_best")
    output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
    list_of_lists, output_filenames = list_of_lists[part_id::num_parts], output_files[part_id::num_parts]
    if 'segmentation_export_params' in trainer.plans.keys():
        force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
        interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
        interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
    else:
        force_separate_z = None
        interpolation_order = 1
        interpolation_order_z = 0
    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(join(dr, f))
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, cleaned_output_files, num_threads_preprocessing,
                                             None)
    all_output_files = []
    do_tta = True
    mixed_precision = True
    torch.cuda.empty_cache()
    for preprocessed in preprocessing:
        output_filename, (d, dct) = preprocessed
        all_output_files.append(all_output_files)
        trainer.load_checkpoint_ram(params[0], False)
        softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
            d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
            step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
            mixed_precision=mixed_precision)[1]
        for p in params[1:]:
            trainer.load_checkpoint_ram(p, False)
            softmax += trainer.predict_preprocessed_data_return_seg_and_softmax(
                d, do_mirroring=do_tta, mirror_axes=trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
                mixed_precision=mixed_precision)[1]
        if len(params) > 1:
            softmax /= len(params)
        transpose_forward = trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get('transpose_backward')
            softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])
        if hasattr(trainer, 'regions_class_order'):
            region_class_order = trainer.regions_class_order
        else:
            region_class_order = None
        results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                          ((softmax, output_filename, dct, interpolation_order, region_class_order,
                                            None, None,
                                            None, None, force_separate_z, interpolation_order_z),)
                                          ))
    _ = [i.get() for i in results]
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
