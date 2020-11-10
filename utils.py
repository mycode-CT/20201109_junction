import glob
import os
from functools import partial

import numpy as np
from monai.data import list_data_collate, CacheDataset
from monai.transforms import (
    Compose,
    LoadNiftid,
    AddChanneld,
    CropForegroundd,
    RandCropByPosNegLabeld,
    Orientationd,
    ToTensord,
    NormalizeIntensityd
)
from scipy.ndimage.measurements import label
from torch import nn
from torch.utils.data import DataLoader

from ellipsoids import create_ellipse_from_mask, nms_ellipsoid, \
    create_list_boxes_fromell


def multi_apply(func, *args, **kwargs):
    '''
    Function to allow multi batch treatment of the different stages of the
    network
    :param func:
    :param args:
    :param kwargs:
    :return:
    '''
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def find_all_positives_and_target(gt_ellipses, image_shape):
    '''
    Functions that return the ellipse maps based on list of ellipses
    characteristics and for each point belonging to an ellipse the shift to
    get to the center and the asociaed lambda, fa and x_vec , y_vec
    :param gt_ellipses:
    :param image_shape:
    :return:
    '''

    ellipse_seg = np.zeros(image_shape)
    delta_x = np.zeros(image_shape)
    delta_y = np.zeros(image_shape)
    delta_z = np.zeros(image_shape)
    lambda_maj = np.zeros(image_shape)
    fa = np.zeros(image_shape)
    x_vec = np.zeros(image_shape)
    y_vec = np.zeros(image_shape)
    x_lin = np.arange(0, image_shape[0])
    y_lin = np.arange(0, image_shape[1])
    z_lin = np.arange(0, image_shape[2])
    x, y, z = np.meshgrid(x_lin, y_lin, z_lin)
    dist = np.ones(image_shape)*1000
    chosen_gt = np.ones(image_shape) * -1

    for (i,f) in enumerate(gt_ellipses):
        #print(i)
        #print(ellipse_seg.max())
        map_ellipse = f.create_mask_fromell(image_shape)
        dist_temp = np.sqrt(np.square(f.x_com-x) + np.square(f.y_com-y) + np.square(f.z_com-z))
        delta_x = np.where(np.logical_and(map_ellipse == 1, dist_temp < dist), f.x_com - x, delta_x)
        lambda_maj = np.where(np.logical_and(map_ellipse == 1, dist_temp < dist), f.lambda_maj, lambda_maj)
        fa = np.where(np.logical_and(map_ellipse == 1, dist_temp < dist), f.fa, fa)
        x_vec = np.where(np.logical_and(map_ellipse == 1, dist_temp < dist), f.x_vec, x_vec)
        y_vec = np.where(np.logical_and(map_ellipse == 1, dist_temp < dist), f.y_vec, y_vec)
        chosen_gt = np.where(np.logical_and(map_ellipse == 1, dist_temp < dist), i, chosen_gt)
        delta_y = np.where(np.logical_and(map_ellipse == 1, dist_temp < dist), f.y_com - y, delta_y)
        delta_z = np.where(np.logical_and(map_ellipse == 1, dist_temp < dist), f.z_com - z, delta_z)
        dist_bis = np.sqrt(np.square(delta_x) + np.square(delta_y) + np.square(delta_z))
        ellipse_seg += map_ellipse
        #print(map_ellipse.max())
        #print(ellipse_seg.max())
        dist = np.where(ellipse_seg == 1, dist_bis, dist)
    # target = np.concat([np.expand_dims(delta_x, 0), np.expand_dims(delta_y, 0),
    #                    np.expand_dims(delta_z, 0), np.expand_dims(lambda_maj,0),
    #                    np.expand_dims(fa, 0), np.expand_dims(x_vec, 0),
    #                    np.expand_dims(y_vec, 0)], 0)
    target = np.concatenate([np.expand_dims(delta_y, 0), np.expand_dims(delta_x, 0),
                             np.expand_dims(delta_z, 0), np.expand_dims(lambda_maj, 0),
                             np.expand_dims(fa, 0), np.expand_dims(x_vec, 0),
                             np.expand_dims(y_vec, 0)], 0)
    #print(ellipse_seg.max())
    ellipse_seg = np.where(ellipse_seg > 0, np.ones_like(ellipse_seg), np.zeros_like(ellipse_seg))
    return ellipse_seg, chosen_gt, target


def sample_ratio_posneg(gt_ellipses, image_shape, ratio=0.5, numb_samples=256):

    '''
    Sampling between positive and negative examples for training of the RPN head
    :param gt_ellipses: list of ellipses from ground truth
    :param image_shape: patch / image shape
    :param ratio: ratio to enforce between positive and negative samples
    :param numb_samples: number of samples to learn from
    :return: Segmented map of ellipses, associated label map (different label
    for different component, target parameter as distance to effective centre of
     ellipsoid and parametric values (lambda_maj, fa, x_vec, y_vec) of the
     ellipsoid, map weight (0 where the samples are not learnt from),
     map weight for the learning of the target ellipsoids (tile in first
     dimension according to number of parameters (7)
    '''
    #image_shape = image_shape.data.cpu().numpy() # chin
    #
    #print(ratio) # chin
    #ratioo = 0.5 # chin
    ellipse_seg, chosen_gt, target = find_all_positives_and_target(gt_ellipses, image_shape)

    # Initialise weights
    weights = np.zeros(image_shape)
    # For each of the gt ellipse sample at least one element and then as many
    #  as required positive samples
    # First check if we have at least the minimum between positive and
    # negative samples and adjust the total number of samples accordingly
    numb_positives = np.sum(ellipse_seg)
    numb_negatives = np.prod(image_shape) - numb_positives
    if numb_positives < ratio * numb_samples or numb_negatives < (1-ratio) * numb_samples:
        max_samples_1 = numb_positives / ratio
        max_samples_2 = numb_negatives / (1-ratio)
        numb_samples_tmp = np.maximum(max_samples_1, max_samples_2) # chin commentted
        #numb_samples = np.maximum(max_samples_1.data.cpu().numpy(), max_samples_2.data.cpu().numpy()) # chin added
        numb_samples = np.minimum(numb_samples, numb_samples_tmp)

    # if numb_positives < ratioo * numb_samples or numb_negatives < (1-ratioo) * numb_samples:
    #     # if numb_positives < ratioo * numb_samples or numb_negatives < (1-ratioo) * numb_samples:
    #     # RuntimeError: bool value of Tensor with more than one value is ambiguous
    #     max_samples_1 = numb_positives / ratioo
    #     max_samples_2 = numb_negatives / (1-ratioo)
    #     #numb_samples = np.maximum(max_samples_1, max_samples_2) # chin commentted
    #     numb_samples = np.maximum(max_samples_1.data.cpu().numpy(), max_samples_2.data.cpu().numpy()) # chin added

    #  sample required number of negative samples
    indices_neg = np.asarray(np.where(ellipse_seg == 0)).T

    chosen_ind = np.random.choice(np.arange(0, indices_neg.shape[0]), np.int(numb_samples * (1-ratio)), replace=False)
    for c in chosen_ind:
        weights[indices_neg[c, 0], indices_neg[c, 1], indices_neg[c, 2]] = 1

    # sample required number of positive samples
    numb_diff = len(np.unique(chosen_gt)) - 1
    values = np.unique(chosen_gt)

    #2020.09.09
    weights_pos = np.zeros_like(ellipse_seg)
    for v in values:
        if v > -1:
            indices_temp = np.asarray(np.where(chosen_gt == v)).T
            numb_available = indices_temp.shape[0]
            expected_samples = numb_samples * ratio / numb_diff
            size_sample = np.minimum(numb_available, expected_samples)
            chosen_ind = np.random.choice(np.arange(0, np.int(indices_temp.shape[0])), np.int(size_sample), replace=False)
            for c in chosen_ind:
                weights[indices_temp[c, 0], indices_temp[c, 1], indices_temp[c, 2]] = 1
                weights_pos[indices_temp[c, 0], indices_temp[c, 1], indices_temp[c, 2]] = 1
    sum_weights = np.sum(weights)
    # print(sum_weights)
    # Continue sampling in case some of the existing label could not provide
    # enough samples
    if sum_weights < numb_samples:
        still_available_pos = ellipse_seg - weights
        still_required = numb_samples - sum_weights
        indices_available = np.asarray(np.where(still_available_pos > 0)).T
        len_available = indices_available.shape[0]
        chosen_ind = np.random.choice(np.arange(0, np.int(indices_available.shape[0])), np.minimum(np.int(len_available), np.int(still_required)), replace=False)
        #chosen_ind = np.random.choice(np.arange(0, indices_available.shape[1]), still_required, replace=False) # chin
        for c in chosen_ind:
            weights[indices_available[c, 0], indices_available[c, 1], indices_available[c, 2]] = 1
    return ellipse_seg, chosen_gt, target, weights, np.tile(np.expand_dims(weights_pos, 0), [target.shape[0], 1, 1, 1])


def create_gt_from_labels(batch_labels):
    '''
    Creating list of ellipsoids from ground truth
    :param batch_labels:
    :return:
    '''
    list_gtbbox_full = []

    for (i, gtseg) in enumerate(batch_labels):
        temp_label, numb_lab = label(np.squeeze(gtseg))
        list_gtbbox = []
        for lab in range(1, numb_lab + 1):
            seg_temp = (temp_label == lab)
            ellipse = create_ellipse_from_mask(seg_temp)
            # # chin add
            # tmp_lambda_maj = ellipse.lambda_maj
            # tmp_para = np.array([ellipse.x_com, ellipse.y_com, ellipse.z_com])
            # if tmp_lambda_maj <= 0:
            #     print("lambda_maj " + str(tmp_lambda_maj))
            # else:
            #     if np.sum(tmp_para <= 0.0) > 0 or np.sum(tmp_para >= 30.0 - 1.0) > 0:  # error I cheated by just simply define patch_sz = 30
            #     #if np.sum(tmp_para <= 0.0) > 0 or np.sum(tmp_para >= patch_sz - 1.0) > 0:  # error
            #         print("tmp_para " + str(tmp_para))
            # del tmp_lambda_maj, tmp_para
            # # end chin add
            list_gtbbox.append(ellipse)

        #list_gtbbox_full.append(torch.from_numpy(np.asarray(list_gtbbox))) #chin commentted
        # TypeError: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, int64, int32, int16, int8, uint8, and bool.
        list_gtbbox_full.append(np.asarray(list_gtbbox)) #chin
    return list_gtbbox_full


def create_proposals(rpn_score, rpn_target, ellipse_seg, chosen_gt, target,
                     nms_ratio, numb_samples=300, ratio=0.5, enforce_gt=True, sphere=7):
    '''
    Still Work in progress: How to create proposals for training purposes
    after run of the RPN - How to apply the NMS, which ratios to use... How
    do we want to focus the training (positive and false positive or still
    using negatives...?
    :param rpn_score:
    :param rpn_target:
    :param ellipse_seg:
    :param chosen_gt:
    :param target:
    :param nms_ratio:
    :param numb_samples:
    :param ratio:
    :param enforce_gt:
    :return:
    '''
    # We could also potentially use the output of the distance regression to
    # help for the selection of boxes to go through the rcnn stage
    sphere_all = np.zeros_like(rpn_target.data.cpu().numpy())
    sphere_all[3, :, :, :] = np.ones((sphere_all.shape[1], sphere_all.shape[2], sphere_all.shape[3]))*7
    #chosen_nms_pos = nms_ellipsoid(rpn_score.data.cpu().numpy(), rpn_target.data.cpu().numpy(), 0.5, nms_ratio) # IndexError: index 452179 is out of bounds for axis 0 with size 4
    #chosen_nms_pos = nms_ellipsoid(rpn_score[1,:,:,:].data.cpu().numpy(), rpn_target.data.cpu().numpy(), threshold_score=0.5, shift=nms_ratio)

    mmt = nn.Softmax(dim=0)
    sm_score = mmt(rpn_score)
    sm_score[:, 0, :, :] = 0
    sm_score[:, rpn_score.shape[1] - 1, :, :] = 0
    sm_score[:, :, rpn_score.shape[2] - 1, :] = 0
    sm_score[:, :, :, rpn_score.shape[3] - 1] = 0
    sm_score[:, :, 0, :] = 0
    sm_score[:, :, :, 0] = 0

    # sm_score = torch.nn.Softmax(rpn_score[1, :, :, :],0)
    # To-DO..please take a good care of the proposal, when there isn't any object.
    chosen_nms_pos_int = sm_score[1,:,:,:].detach().numpy()
    #threshold_score_pos = 0.95*chosen_nms_pos_int.max()
    threshold_score_pos = 0.95 * chosen_nms_pos_int.max()
    #threshold_score_pos = 0.9 * chosen_nms_pos_int.max() # chin chose 0.95 himself
    chosen_nms_pos = nms_ellipsoid(sm_score[1,:,:,:].detach().numpy(), rpn_target.detach().numpy(), threshold_score=threshold_score_pos, shift=nms_ratio)
    #chosen_nms_neg = nms_ellipsoid(1-rpn_score.data.cpu().numpy(), sphere_all, threshold_score=0.75, shift=nms_ratio) # IndexError: index 902587 is out of bounds for axis 0 with size 4
    #chosen_nms_neg_int = 1 - sm_score[1,:,:,:].detach().numpy()
    chosen_nms_neg_int = sm_score[0, :, :, :].detach().numpy()
    #threshold_score_neg = 0.95*chosen_nms_neg_int.max()
    #threshold_score_neg = 0.95 * chosen_nms_neg_int.max()
    threshold_score_neg = 0.9995 * chosen_nms_neg_int.max() # chin chose 0.25 himself
    chosen_nms_neg = nms_ellipsoid(sm_score[0, :, :, :].detach().numpy(), sphere_all, threshold_score=threshold_score_neg, shift=nms_ratio)
    gt_ellipses = []
    if enforce_gt:
        list_gt = np.unique(chosen_gt)
        for l in list_gt:
            if l > -1:
                #seg_temp = np.where(chosen_gt == l)
                seg_temp = np.where(chosen_gt == l, np.ones_like(ellipse_seg), np.zeros_like(ellipse_seg))
                ell_mask = ellipse_seg * seg_temp
                ellipse_gt = create_ellipse_from_mask(ell_mask)
                #ellipse_gt = create_ellipse_from_mask(ellipse_seg[chosen_gt==l]) # IndexError: index 1 is out of bounds for axis 0 with size 1
                gt_ellipses.append(ellipse_gt)

    numb_needed = numb_samples - len(gt_ellipses)
    numb_pos = len(chosen_nms_pos)
    numb_neg = len(chosen_nms_neg)
    if numb_pos < ratio * numb_needed:
        numb_needed = numb_pos / ratio
    if numb_neg < (1-ratio) * numb_needed:
        numb_needed = numb_neg / (1-ratio)
    numb_pos = ratio * numb_needed
    numb_neg = (1-ratio) * numb_needed
    list_pos = chosen_nms_pos[:int(numb_pos)]
    list_neg = chosen_nms_neg[:int(numb_neg)]
    list_all = gt_ellipses + list_pos + list_neg
    list_boxes, list_valid = create_list_boxes_fromell(list_all, ellipse_seg.shape)

    # Create associated list of targets
    list_label = []
    list_target = []
    list_weight = []

    for (e, v) in zip(list_all, list_valid):
        if v == 1:
            list_label.append(ellipse_seg[int(e.x_com), int(e.y_com), int(e.z_com)])
            list_target.append(np.expand_dims(target[:, int(e.x_com), int(e.y_com), int(e.z_com)], axis=-1))
            #list_label.append(ellipse_seg[int(e.y_com), int(e.x_com), int(e.z_com)])
            #list_target.append(np.expand_dims(target[:, int(e.y_com), int(e.x_com), int(e.z_com)], axis=-1))
            list_weight.append(1)

    # for (e, v) in zip(list_all, list_valid):
    #     if v == 1:
    #         list_label.append([int(e.x_com), int(e.y_com), int(e.z_com)])
    #         list_target.append(np.expand_dims(target[:, int(e.x_com), int(e.y_com), int(e.z_com)], axis=-1))
    #         list_weight.append(1)

    #rcnn_lab = np.zeros(len(list_all))
    #rcnn_lab[:len(gt_ellipses) + int(numb_pos)] = 1
    #print(len(list_label))
    #print(len(list_boxes))
    #return rcnn_lab, list_target, list_label, list_boxes
    return list_label, list_weight, list_target, list_boxes, list_pos, list_neg

    ## Notes
    # rcnn_lab -> rcnn_lab (ones*size(gt_ellepes)
    # -> rcnn_lab_weights
    # list_target -> rcnn_targets
    # list_label -> rcnn_weights_bbox
    # list_boxes -> nms_boxes


    # Output at skeleton:  rcnn_lab, rcnn_lab_weights, rcnn_targets, rcnn_weights_bbox, nms_boxes,

    # positives = np.where(rpn_score>=0.5,np.ones_like(rpn_score),
    #                      np.zeros_like(rpn_score))
    # negatives = np.where(rpn_score<0.5, np.ones_like(rpn_score),
    #                      np.zeros_like(rpn_score))
    # true_positives = positives * ellipse_seg
    # true_negatives = negatives * (1-ellipse_seg)
    # false_positives = positives * (1-ellipse_seg)
    # false_negatives = negatives * ellipse_seg
    #
    # # We need to learn from each of the examples
    # numb_diff = len(np.unique(chosen_gt*positives))-1

def ensuring_large_enough_boxes(list_box, min_size, shape):
    list_box_new = []
    for b in list_box:
        b_new = b#np.zeros_like(b)
        for d in range(0,3):
            #diff = min_size[d] - (b[3+d]-b[d])
            diff = min_size[d] - (b[3+d]-b[d])
            if diff > 0:
                #half_diff = int(diff/2.0)
                half_diff = int(diff / 2.0) + 1
                b_new[d] = np.maximum(0, b[d]-half_diff)
                b_new[3+d] = np.minimum(shape[d]-1,b[3+d]+half_diff)
        list_box_new.append(b_new)
    return list_box_new


def get_dataloader(train_root, batch_size, patch_size, num_samples, val_root=None):
    """ Function to get training and validation data loaders. """

    train_images = sorted(glob.glob(os.path.join(train_root, 'OP', '*.nii.gz')))
    train_labels = sorted(glob.glob(os.path.join(train_root, 'Labels', 'DistanceMap_*.nii.gz')))
    train_bilabels = sorted(glob.glob(os.path.join(train_root, 'Labels_binary', 'Bi_DistanceMap_*.nii.gz')))

    train_files = [
        {
            'image': image_name,
            'label': label_name,
            'bilabel': bilabel_name
        } for image_name, label_name, bilabel_name in zip(train_images, train_labels, train_bilabels)
    ]

    train_files = train_files[:]

    train_transforms = Compose([
        LoadNiftid(keys=['image', 'label', 'bilabel']),
        AddChanneld(keys=['image', 'label', 'bilabel']),
        Orientationd(keys=['image', 'label', 'bilabel'], axcodes='RAS'),
        NormalizeIntensityd(keys=['image']),
        # RandGaussianNoised(keys=['image'], prob=0.75, mean=0.0, std=1.75),
        # RandRotate90d(keys=['image', 'label', 'bilabel'], prob=0.5, spatial_axes=[0, 2]),
        CropForegroundd(keys=['image', 'label', 'bilabel'], source_key='image'),
        # RandCropByPosNegLabeld(
        #     keys=['image', 'label', 'bilabel'], label_key='bilabel', size=[patch_sz, patch_sz, patch_sz], pos=1, neg=1, num_samples=n_samp
        # ),
        RandCropByPosNegLabeld(
            keys=['image', 'label', 'bilabel'],
            label_key='bilabel',
            size=[patch_size, patch_size, patch_size],
            pos=1, neg=1,
            num_samples=num_samples, image_threshold=1
        ),
        # Spacingd(keys=['image', 'label', 'bilabel'], pixdim=(new_spacing, new_spacing, new_spacing), interp_order=(2, 0), mode='nearest'),
        ToTensord(keys=['image', 'label', 'bilabel'])
    ])

    ## Define CacheDataset and DataLoader for training and validation
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)

    # Suggestion:
    # According to https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813
    # a num_worker = 4 * num_GPU is a good heuristic to follow
    # Also, when working with GPUs, it is recommended to use pin_memory=True to store cache in the GPUs
    # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              collate_fn=list_data_collate)

    if val_root:
        val_images = sorted(glob.glob(os.path.join(val_root, 'OP', '*.nii.gz')))
        val_labels = sorted(glob.glob(os.path.join(val_root, 'Labels', 'DistanceMap_*.nii.gz')))
        val_bilabels = sorted(glob.glob(os.path.join(val_root, 'Labels_binary', 'Bi_DistanceMap_*.nii.gz')))
        # val_bilabels = sorted(glob.glob(os.path.join(val_root, 'newLabels', 'SegmentedPerSlice_SABRE_*.nii.gz')))
        val_files = [{'image': image_name, 'label': label_name, 'bilabel': bilabel_name} for
                     image_name, label_name, bilabel_name in zip(val_images, val_labels, val_bilabels)]

        val_files = val_files[:]

        val_transforms = Compose([
            LoadNiftid(keys=['image', 'label', 'bilabel']),
            AddChanneld(keys=['image', 'label', 'bilabel']),
            Orientationd(keys=['image', 'label', 'bilabel'], axcodes='RAS'),
            NormalizeIntensityd(keys=['image']),
            CropForegroundd(keys=['image', 'label', 'bilabel'], source_key='image'),
            RandCropByPosNegLabeld(
                keys=['image', 'label', 'bilabel'],
                label_key='bilabel',
                size=[patch_size, patch_size, patch_size],
                pos=1, neg=1,
                num_samples=num_samples, image_threshold=1
            ),
            ToTensord(keys=['image', 'label', 'bilabel'])
        ])
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                collate_fn=list_data_collate)

        return train_loader, val_loader

    return train_loader
