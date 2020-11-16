import torch
from torch.autograd import Variable
from torch import optim
import numpy as np
from torch import nn
import torchvision
from losses import smooth_l1_loss, smooth_l2_loss, soft_dice_loss, \
    weighted_soft_dice_loss, focal_loss, cross_entropy
from nnUnet import nnUnet
from networks import RPNHead, BBoxHead, BBoxHead_Conv
from utils import multi_apply, find_all_positives_and_target, \
    sample_ratio_posneg, create_gt_from_labels, create_proposals, ensuring_large_enough_boxes
from ellipsoids import create_ellipse_from_mask, create_list_ellipses, correct_ellipses_list
from ellipsoids import Ellipsoid, nms_ellipsoid, create_list_boxes_fromell
from tqdm import tqdm
import sys
import time
import datetime
import nibabel as nib

## From MONAI code
import os
import sys
import glob
from torch.utils.data import DataLoader
import matplotlib #chin
import matplotlib.pyplot as plt
import monai
from monai.transforms import \
    Compose, LoadNiftid, AddChanneld, ScaleIntensityRanged, CropForegroundd, \
    RandCropByPosNegLabeld, RandAffined, Spacingd, Orientationd, RandSpatialCrop, ToTensord, AsChannelFirstd, ScaleIntensityd, RandRotate90d, NormalizeIntensityd, RandGaussianNoised
from monai.data import list_data_collate, sliding_window_inference
from monai.networks.layers import Norm
from monai.metrics import compute_meandice
from scipy.ndimage.measurements import label
#import gc
monai.config.print_config()

pwd = os.getcwd()
#pwd = '/nfs/home/ctangwiriyasakul/DGXwork/projects/202011_nov/week03/20201116_junction/beta_3train_DGX_syn'
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
## End of setting from MONAI

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Const_strcData:
    pass

class TwoStageDetector(nn.Module):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 rcnn_head=None,
                 bbox_head=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()

        self.backbone = backbone

        if rpn_head is not None:
            self.rpn_head = rpn_head
        if bbox_head is not None:
            self.bbox_head = bbox_head
        if rcnn_head is not None:
            self.bbox_roi_extractor = bbox_roi_extractor
            self.rcnn_head = rcnn_head

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'rcnn_head') and self.rcnn_head is not None

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            if 'backbone' in pretrained.keys():
                # self.backbone.load_state_dict(torch.load(pretrained['backbone'], map_location=torch.device(device)))
                self.load_state_dict(torch.load(pretrained['backbone'], map_location=torch.device(device)))
            if 'rpn_head' in pretrained.keys():
                # self.rpn_head.load_state_dict(torch.load(pretrained['rpn_head'], map_location=torch.device(device)))
                self.load_state_dict(torch.load(pretrained['rpn_head'], map_location=torch.device(device)))
            if 'rcnn_head' in pretrained.keys():
                # self.rcnn_head.load_state_dict(torch.load(pretrained['rcnn_head'], map_location=torch.device(device)))
                self.load_state_dict(torch.load(pretrained['rcnn_head'], map_location=torch.device(device)))
            if 'full' in pretrained.keys():
                test = torch.load(pretrained['full'], map_location=torch.device(device))
                self.load_state_dict(torch.load(pretrained['full'], map_location=torch.device(device)))

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        feat, x = self.backbone(img)
        print("Outside: image size", img.size(), "feat_size", feat.size(), "x_size", x.size())
        #return list_feat[-1], list_feat[0]
        return feat, x # chin

    def inference_rpn(self, img):
        feat, x = self.extract_feat(img)
        # rpn_cls, rpn_ell, _ = self.rpn_head(feat)
        rpn_cls, rpn_bbox, _ = self.rpn_head(feat)

        ## force fa and lambda_maj
        # m_tmp = nn.Sigmoid()
        # rpn_bbox[:, 4, ...] = m_tmp(rpn_bbox[:, 4, ...])  # fa
        # position_neg = torch.where(rpn_bbox[:, 3, :, :, :] < 0)  # lambda_maj
        # rpn_bbox[:, 3, position_neg[1], position_neg[2], position_neg[3]] = torch.abs(
        #     rpn_bbox[:, 3, position_neg[1], position_neg[2], position_neg[3]])
        rpn_bbox = rpn_bbox.type(torch.DoubleTensor)
        ## Enf of force fa and lambda_maj

        # print("Assignment RPN done")
        scores = torch.nn.functional.softmax(rpn_cls, dim=1)[:, 1, :, :, :]
        selected_ell = torch.zeros_like(rpn_bbox)
        for ij in range(selected_ell.shape[0]):
            tmp_rpn_bbox = rpn_bbox[ij, :, :, :, :]
            tmp_scores = scores[ij, :, :, :]
            for ik in range(7):
                tmp_rpn_bbox2 = tmp_rpn_bbox[ik, :, :, :]
                tmp_rpn_bbox3 = torch.zeros_like(tmp_rpn_bbox2)
                selected_ell[ij,ik,:,:,:] = torch.where(tmp_scores > 0.5, tmp_rpn_bbox2, tmp_rpn_bbox3)
        #scores = scores.type(torch.DoubleTensor) # chin
        #selected_ell = torch.from_numpy(selected_ell)
        # TODO Could use hellinger distance to perform 3d nms - For the moment
        detached_scores = scores.detach().numpy()
        detached_ell = selected_ell.detach().numpy()
        #nms_proposals = nms_ellipsoid(np.squeeze(scores.detach().numpy()), np.squeeze(selected_ell.detach().numpy()), shift=False, threshold_score=0.5,threshold_overlap=0.2) # need to check with carole if she want it to be false
        if selected_ell.sum() > 0:
            nms_proposals = []
            for b in range(0, scores.shape[0]):
                if detached_ell[b, ...].sum() > 0:
                    ell_tmp = nms_ellipsoid(detached_scores[b, ...], detached_ell[b, ...], shift=False, threshold_score=0.5, threshold_overlap=0.2)
                    nms_proposals.append(ell_tmp)
                else:
                    nms_proposals.append([])
        else:
            nms_proposals = []
            nms_proposals.append([])
        return feat, nms_proposals, rpn_bbox, selected_ell, scores

    def inference_rcnn(self, img):
        feat, nms_proposals, rpn_bbox, selected_ell, scores = self.inference_rpn(img)  # chin
        list_box = []
        for b in range(0, len(nms_proposals)):
            list_box_tmp, list_valid_tmp = create_list_boxes_fromell(nms_proposals[b], img.shape[2:])
            list_box.append(list_box_tmp)

        list_extracted = []
        for (b, list_box2) in enumerate(list_box):
            list_extracted2 = []
            print(list_box2)
            for (b3, list_box3) in enumerate(list_box2):
                list_box_new = ensuring_large_enough_boxes([list_box3], [7, 7, 7], img.shape[2:])
                for bb in list_box_new:
                    #extracted = x[b, list_box_new[0]:list_box_new[3],list_box_new[1]:list_box_new[4],list_box_new[2]:list_box_new[5]]
                    extracted = feat[b3:b3+1, :, bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]
                    list_extracted2.append(extracted)
            list_extracted.append(list_extracted2)


        #     list_box_new = ensuring_large_enough_boxes(list_box2, [7, 7, 7], img.shape[2:])
        #     list_extracted2 = []
        #     for bb in list_box_new:
        #         # extracted = x[b, list_box_new[0]:list_box_new[3],list_box_new[1]:list_box_new[4],list_box_new[2]:list_box_new[5]]
        #         extracted = feat[b:b + 1, :, bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]
        #         list_extracted2.append(extracted)
        #     list_extracted.append(list_extracted2)
        #
        # list_extracted = []
        # for (b, list_box) in enumerate(list_box):
        #     list_box_new = ensuring_large_enough_boxes(list_box, [7, 7, 7], img.shape[2:])
        #     for bb in list_box_new:
        #         # extracted = x[b, list_box_new[0]:list_box_new[3],list_box_new[1]:list_box_new[4],list_box_new[2]:list_box_new[5]]
        #         extracted = feat[b:b + 1, :, bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]
        #         list_extracted.append(extracted)

        list_inf_ell_rcnn = []
        list_inf_cls_rcnn = []
        for (a, ext) in enumerate(list_extracted):
            list_inf_ell_rcnn2 = []
            list_inf_cls_rcnn2 = []
            for (a2, ext2) in enumerate(ext):
                final_cls, final_reg = self.rcnn_head(ext2)
                list_inf_cls_rcnn2.append(final_cls)
                list_inf_ell_rcnn2.append(final_reg)
            list_inf_cls_rcnn.append(list_inf_cls_rcnn2)
            list_inf_ell_rcnn.append(list_inf_ell_rcnn2)

        m33 = nn.Softmax(dim=1)

        list_pos_cls_rcnn = []
        list_pos_ell_rcnn = []
        for (b, list_inf_cls_rcnn_ext) in enumerate(list_inf_cls_rcnn):
            list_pos_cls_rcnn2 = []
            list_pos_ell_rcnn2 = []
            for (b2, ext22) in enumerate(list_inf_cls_rcnn_ext):
                if len(list_inf_cls_rcnn[b][b2]) > 0: # check if it is not an empty matrix.
                    tmpB = list_inf_cls_rcnn[b][b2]
                    tmpB = m33(tmpB)  # [:, 1] #chin
                    tmpC = list_inf_ell_rcnn[b][b2]
                    if np.squeeze(tmpB)[1] > 0.5:#0.499999999999:
                        print('pass 0.5')
                        list_pos_cls_rcnn2.append(tmpB)
                        list_pos_ell_rcnn2.append(tmpC)
                    print('chin2')
                else:
                    print('it is an empty array')
                    print('chin2')
            list_pos_cls_rcnn.append(list_pos_cls_rcnn2)
            list_pos_ell_rcnn.append(list_pos_ell_rcnn2)

        list_new_ell_rcnn = list_pos_ell_rcnn
        for (b, list_new_ell_rcnn_ext) in enumerate(list_new_ell_rcnn):
            for (b2, list_new_ell_rcnn_ext2) in enumerate(list_new_ell_rcnn_ext):
                tmpE2 = np.squeeze(list_new_ell_rcnn[b][b2].data.cpu().numpy())
                tmpE2[0] = tmpE2[0] + nms_proposals[b][b2].x_com
                tmpE2[1] = tmpE2[1] + nms_proposals[b][b2].y_com
                tmpE2[2] = tmpE2[2] + nms_proposals[b][b2].z_com
                tmpE2[3] = tmpE2[3] + nms_proposals[b][b2].lambda_maj
                tmpE2[4] = tmpE2[4] + nms_proposals[b][b2].fa
                tmpE2[5] = tmpE2[5] + nms_proposals[b][b2].x_vec
                tmpE2[6] = tmpE2[6] + nms_proposals[b][b2].y_vec
                list_new_ell_rcnn[b][b2][0][0] = torch.tensor(tmpE2[0])
                list_new_ell_rcnn[b][b2][0][1] = torch.tensor(tmpE2[1])
                list_new_ell_rcnn[b][b2][0][2] = torch.tensor(tmpE2[2])
                list_new_ell_rcnn[b][b2][0][3] = torch.tensor(tmpE2[3])
                list_new_ell_rcnn[b][b2][0][4] = torch.tensor(tmpE2[4])
                list_new_ell_rcnn[b][b2][0][5] = torch.tensor(tmpE2[5])
                list_new_ell_rcnn[b][b2][0][6] = torch.tensor(tmpE2[6])

        #AA = list_new_ell_rcnn  # change this later
        image_final = np.zeros_like(img)
        inf_ellipese = []
        for (ll, fin_ell) in enumerate(list_new_ell_rcnn):
            image_final2 = np.zeros_like(img[0,0,...])
            inf_ellipese2 = []
            for (ll2, fin_ell2) in enumerate(fin_ell):
                #ll_chin = Ellipsoid(x_com=fin_ell[ll2][0][0].detach().numpy(), y_com=fin_ell[ll2][0][1].detach().numpy(), z_com=fin_ell[ll2][0][2].detach().numpy(),
                #                    lambda_maj=fin_ell[ll2][0][3].detach().numpy(), fa=fin_ell[ll2][0][4].detach().numpy(), x_vec=fin_ell[ll2][0][5].detach().numpy(),
                #                    y_vec=fin_ell[ll2][0][6].detach().numpy())
                # cheat to swopt x_com and y_com
                ll_chin = Ellipsoid(x_com=fin_ell[ll2][0][1].detach().numpy(), y_com=fin_ell[ll2][0][0].detach().numpy(), z_com=fin_ell[ll2][0][2].detach().numpy(),
                                    lambda_maj=fin_ell[ll2][0][3].detach().numpy(), fa=fin_ell[ll2][0][4].detach().numpy(), x_vec=fin_ell[ll2][0][5].detach().numpy(),
                                    y_vec=fin_ell[ll2][0][6].detach().numpy())
                #inf_ellipese2.append(np.asarray(ll_chin))
                inf_ellipese2.append(ll_chin)
                mask_new = ll_chin.create_mask_fromell(img.detach().numpy().shape[2:]) # shape of patch
                image_final2 += mask_new
                print(image_final2.sum())
            #inf_ellipese.append(inf_ellipese2)
            inf_ellipese.append(np.asarray(inf_ellipese2))  # chin
            print(ll)
            image_final[ll,0,...] = image_final2
            print(image_final.sum())
        image_final = torch.tensor(image_final)
        print(image_final.sum())

        return image_final, list_new_ell_rcnn, inf_ellipese, feat, nms_proposals, rpn_bbox, selected_ell, scores

    def forward_train(self,
                      img,
                      gt_ellipses=None,
                      image_shape=None,
                      gt_labels=None,
                      label_seg=None,
                      dist_seg=None,
                      training_stage=None#['rpn_head'] 20200920 chin editted
                      ):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_shape .

            gt_ellipses (list[list of Ellipses]): each item are the truth
            ellipse for each image in Ellipsoid format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            dist_seg: image with the distance map to regress as part of the
            backbone training.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            training_stage : list of training steps to consider among
            backbone, rpn_head, rcnn_head.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        feat, x = self.extract_feat(img) # chin
        # freeze_model(self.backbone)
        if 'backbone' in training_stage and dist_seg is not None:
            STD_torch_lossBB = nn.L1Loss() # Chin
            ##STD_torch_lossBB = nn.MSELoss()
            losses['backbone'] = STD_torch_lossBB(torch.sigmoid(x), dist_seg)  # Chin
            #print(losses['backbone'])
            losses['rpn_cls'] = 0
            losses['rpn_bbox'] = 0
            losses['rcnn_cls'] = 0
            losses['rcnn_bbox'] = 0
        if 'rpn_head' in training_stage or 'rcnn_head' in training_stage:
            #print(training_stage)
            #print("running RPN")
            # Creating weights or select points for RPN
            img_shape_full = [img[i].shape[-3:] for i in range(0, len(gt_ellipses))] # Chin make it 3D
            rpn_cls, rpn_bbox, _ = self.rpn_head(feat)  ## not sure what this means

            #print("RPN performed")
            (maps_pos, attributed_pos, bbox_targets, weights, bbox_weights) = \
                multi_apply(sample_ratio_posneg,
                                          gt_ellipses,
                                          img_shape_full)

            rpn_prob = torch.nn.functional.softmax(rpn_cls, dim=1)[:, 1, :, :, :]

            list_map = [np.expand_dims(m, 0) for m in maps_pos]
            list_weights = [np.expand_dims(m, 0) for m in weights]
            labels_tmp = np.concatenate(list_map, 0)
            #labels_tmp[labels_tmp == 2] = 1 #chin cheat NEED TO ASK Carole to correct
            weights_tmp = np.concatenate(list_weights, 0)
            labels_tmp = torch.from_numpy(labels_tmp)
            weights_tmp = torch.from_numpy(weights_tmp)
            labels_fin = labels_tmp.type(torch.FloatTensor)
            label_weights_fin = weights_tmp.type(torch.FloatTensor)
            labels_fin = labels_fin.type(torch.LongTensor)

            rpn_cls = rpn_cls.to(device)
            labels_fin = labels_fin.to(device)
            label_weights_fin = label_weights_fin.to(device)

            rpn_prob = rpn_prob.to(device)
            labels_fin = labels_fin.to(device)
            label_weights_fin = label_weights_fin.to(device)

            loss_cls_focal = focal_loss(rpn_cls,
                                        targets=labels_fin,
                                        weights=Variable(label_weights_fin, requires_grad=True))

            loss_cls_dice = weighted_soft_dice_loss(rpn_prob,
                                                    labels_fin,
                                                    Variable(label_weights_fin, requires_grad=True))

            # loss_cls_focal = focal_loss(rpn_cls,
            #                             targets=labels_fin.type(torch.LongTensor),
            #                             weights=Variable(label_weights_fin, requires_grad=True))
            # loss_cls_dice = weighted_soft_dice_loss(rpn_prob,
            #                                         labels_fin,
            #                                         Variable(label_weights_fin, requires_grad=True))
            # loss_cls.requires_grad = True
            loss_cls = loss_cls_dice + loss_cls_focal
            #print(loss_cls_dice, loss_cls_focal)

            bbox_targets_tmp = np.zeros((rpn_bbox.shape[0], rpn_bbox.shape[1], rpn_bbox.shape[2], rpn_bbox.shape[3], rpn_bbox.shape[4]))
            bbox_weights_tmp = np.zeros((rpn_bbox.shape[0], rpn_bbox.shape[1], rpn_bbox.shape[2], rpn_bbox.shape[3], rpn_bbox.shape[4]))
            for k in range(len(bbox_targets)):
                bbox_targets_tmp[k,:,:,:,:] = bbox_targets[k]
                bbox_weights_tmp[k,:,:,:,:] = bbox_weights[k]

            bbox_targets_tmp = torch.from_numpy(bbox_targets_tmp)
            bbox_weights_tmp = torch.from_numpy(bbox_weights_tmp)

            #loss_rpnbb = smooth_l1_loss(rpn_bbox, bbox_targets_tmp, bbox_weights_tmp)

            STD_torch_loss = nn.L1Loss()
            # STD_torch_loss = nn.MSELoss()

            bbox_weights_tmp = bbox_weights_tmp.type(torch.LongTensor)

            rpn_bbox = rpn_bbox.to(device)
            bbox_weights_tmp = bbox_weights_tmp.to(device)
            bbox_targets_tmp = bbox_targets_tmp.to(device)

            # loss_rpnbb = STD_torch_loss(rpn_bbox*bbox_weights_tmp, bbox_targets_tmp*bbox_weights_tmp.type(torch.LongTensor))
            loss_rpnbb = STD_torch_loss(rpn_bbox * bbox_weights_tmp, bbox_targets_tmp * bbox_weights_tmp)

            # del bbox_targets_tmp, bbox_weights_tmp

            # loss_rpnbb = smooth_l1_loss(rpn_bbox, torch.stack(torch.from_numpy(bbox_targets)),torch.stack(bbox_weights).reshape([len(gt_ellipses), 7, image_shape[0],image_shape[1],image_shape[2]]))
            # loss_rpnbb.requires_grad = True
            losses['rpn_cls'] = loss_cls
            losses['rpn_bbox'] = loss_rpnbb

            losses['rcnn_cls'] = 0
            losses['rcnn_bbox'] = 0
            #print("Loss RPN calculated")
        if 'rcnn_head' in training_stage:
            rcnn_lab, rcnn_weights, rcnn_targets, rcnn_boxes, rcnn_pos, rcnn_neg = multi_apply(create_proposals,
                                                                                               rpn_cls,
                                                                                               rpn_bbox,
                                                                                               maps_pos,
                                                                                               attributed_pos,
                                                                                               bbox_targets,
                                                                                               (False,) * len(
                                                                                                   gt_ellipses))
            # print("NMS performed for proposal")

            rcnn_lab2 = []
            rcnn_weights2 = []

            for ii in range(len(rcnn_lab)):
                tmp_rcnn_lab = np.array(rcnn_lab[ii])
                rcnn_lab2 = np.concatenate((rcnn_lab2, tmp_rcnn_lab), axis=0)
                tmp_rcnn_weights = np.array(rcnn_weights[ii])
                rcnn_weights2 = np.concatenate((rcnn_weights2, tmp_rcnn_weights), axis=0)
                del tmp_rcnn_lab, tmp_rcnn_weights

            rcnn_targets2 = np.zeros((rcnn_lab2.shape[0], 7))
            countt = 0
            for ii in range(len(rcnn_lab)):
                rcnn_targets2_2 = []
                for jj in range(len(rcnn_targets[ii])):
                    tmp_rcnn_targets2 = np.array(rcnn_targets[ii][jj])
                    rcnn_targets2[countt, :] = np.reshape(tmp_rcnn_targets2, (tmp_rcnn_targets2.shape[0],))
                    countt = countt + 1
                del rcnn_targets2_2

            rcnn_lab = torch.from_numpy(rcnn_lab2)
            rcnn_lab_weights = torch.from_numpy(rcnn_weights2)
            rcnn_targets = torch.from_numpy(rcnn_targets2)

            del rcnn_lab2, rcnn_weights2, rcnn_targets2

            list_extracted = []
            for (b, list_box) in enumerate(rcnn_boxes):
                list_box_new = ensuring_large_enough_boxes(list_box, [7, 7, 7], np.shape(rpn_cls)[2:])
                for bb in list_box_new:
                    # extracted = x[b, list_box_new[0]:list_box_new[3],list_box_new[1]:list_box_new[4],list_box_new[2]:list_box_new[5]]
                    extracted = feat[b:b + 1, :, bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]
                    list_extracted.append(extracted)
            # print("Extraction performed", aligned_feat.shape)

            # 2020.08.05
            # Apply RCNN head to ROIs
            rcnn_cls = []
            rcnn_ell = []
            rcnn_mask = []
            for e in list_extracted:
                cls_temp, ell_temp = self.rcnn_head(e)
                rcnn_cls.append(cls_temp)
                rcnn_ell.append(ell_temp)

            # if len(rcnn_cls) == 0:
            #     tmp_rcnn_cls = torch.Tensor([0.0])
            #     rcnn_cls.append(tmp_rcnn_cls)
            #     #rcnn_cls_fin = torch.cat(rcnn_cls)
            #     #rcnn_cls_fin = rcnn_cls_fin.type(torch.FloatTensor)

            if len(rcnn_ell) != 0:
                rcnn_ell_fin = torch.cat(rcnn_ell)
                rcnn_ell_fin = rcnn_ell_fin.type(torch.FloatTensor)
            else:
                print('warning RCNN ell')
                tmp_rcnn = torch.zeros((rcnn_targets.shape[0], 7))#torch.Tensor([0, 0, 0, 0, 0, 0, 0])
                rcnn_ell.append(tmp_rcnn)
                rcnn_ell_fin = torch.cat(rcnn_ell)
                rcnn_ell_fin = rcnn_ell_fin.type(torch.FloatTensor)

            if len(rcnn_cls) > 0:
                rcnn_cls2 = rcnn_cls[0]
                for ij in range(1, len(rcnn_cls)):
                    rcnn_cls2 = torch.cat((rcnn_cls2, rcnn_cls[ij]), 0)
                m22 = nn.Softmax(dim=1)
                # rcnn_prob = m22(rcnn_cls2)#[:, 1]
                rcnn_prob = m22(rcnn_cls2)  # [:, 1] #chin
            else:
                print('warning RCNN cls')
                #tmp_rcnn_cls = torch.Tensor([0.0]) #chin cheated
                rcnn_prob = torch.zeros((rcnn_lab.shape[0],2))

            # loss_rcnn_cls_dice = weighted_soft_dice_loss(rcnn_prob,
            #                                         rcnn_lab,
            #                                         rcnn_lab_weights)
            # loss_rcnn_cls_ce = focal_loss(rcnn_cls, rcnn_lab,
            #                               weights=rcnn_lab_weights, gamma=2)

            rcnn_prob = rcnn_prob.type(torch.FloatTensor)
            rcnn_lab_weights = rcnn_lab_weights.type(torch.FloatTensor)

            print('target=', rcnn_lab.sum())

            loss_rcnn_cls_ce = focal_loss(rcnn_prob, rcnn_lab.type(torch.LongTensor), weights=rcnn_lab_weights, gamma=2)
            # loss_rcnn_cls_ce = focal_loss(Variable(rcnn_prob, requires_grad = True), rcnn_lab.type(torch.LongTensor), weights=Variable(rcnn_lab_weights, requires_grad=True), gamma=1)
            # loss_rcnn_cls_ce = focal_loss(Variable(rcnn_prob, requires_grad=True), rcnn_lab.type(torch.LongTensor), weights=Variable(rcnn_lab_weights), gamma=2)
            # loss_rcnn_cls_ce = focal_loss(Variable(rcnn_prob, requires_grad=True), rcnn_lab.type(torch.LongTensor))
            loss_rcnn_cls = loss_rcnn_cls_ce

            STD_torch_lossRCNN = nn.L1Loss()
            #loss_rcnn_bb = STD_torch_lossRCNN(rcnn_ell_fin,rcnn_targets.type(torch.LongTensor))
            loss_rcnn_bb = STD_torch_lossRCNN(rcnn_ell_fin.type(torch.FloatTensor),rcnn_targets.type(torch.FloatTensor))

            losses['rcnn_cls'] = loss_rcnn_cls
            losses['rcnn_bbox'] = loss_rcnn_bb

            del loss_rcnn_cls, loss_rcnn_bb, loss_rcnn_cls_ce

        # if 'rpn_head' not in training_stage:
        #     freeze_model(self.rpn_head)
        # if 'rcnn_head' not in training_stage:
        #     freeze_model(self.rcnn_head)

        return losses

def chin_select_data(keep_matrix, patch_sz, batch_images_in, batch_labels_in, batch_bilabels_in):
    batch_images_out = np.zeros((len(keep_matrix), 1, patch_sz, patch_sz, patch_sz))
    batch_labels_out = np.zeros((len(keep_matrix), 1, patch_sz, patch_sz, patch_sz))
    batch_bilabels_out = np.zeros((len(keep_matrix), 1, patch_sz, patch_sz, patch_sz))

    for ii in range(len(keep_matrix)):
        batch_images_out[ii, 0, :, :, :] = batch_images_in[keep_matrix[ii], 0, :, :, :]
        batch_labels_out[ii, 0, :, :, :] = batch_labels_in[keep_matrix[ii], 0, :, :, :]
        batch_bilabels_out[ii, 0, :, :, :] = batch_bilabels_in[keep_matrix[ii], 0, :, :, :]

    batch_images_out = batch_images_out.astype(dtype='float32')
    batch_images_out = torch.from_numpy(batch_images_out)
    # batch_images_out = batch_images_out.to(device)

    batch_labels_out = batch_labels_out.astype(dtype='float32')
    batch_labels_out = torch.from_numpy(batch_labels_out)
    # batch_labels_out = batch_images_out.to(device)

    batch_bilabels_out = batch_bilabels_out.astype(dtype='float32')
    batch_bilabels_out = torch.from_numpy(batch_bilabels_out)
    return batch_images_out, batch_labels_out, batch_bilabels_out

def chin_save_Ellipes(save_fold, gt_ellipses):
    GT_para = Const_strcData()
    GT_para.lambda_maj = []
    GT_para.x_com = []
    GT_para.y_com = []
    GT_para.z_com = []
    GT_para.fa = []
    GT_para.x_vec = []
    GT_para.y_vec = []
    for i in range(len(gt_ellipses)):
        # tmp_lambda_maj = gt_ellipses_tmp[i][0].lambda_maj
        # tmp_lambda_maj = gt_ellipses_tmp[i]
        keep_or_not = 0  # 0 means keep
        for j in range(len(gt_ellipses[i])):
            GT_para.lambda_maj.append(gt_ellipses[i][j].lambda_maj)
            GT_para.x_com.append(gt_ellipses[i][j].x_com)
            GT_para.y_com.append(gt_ellipses[i][j].y_com)
            GT_para.z_com.append(gt_ellipses[i][j].z_com)
            GT_para.fa.append(gt_ellipses[i][j].fa)
            GT_para.x_vec.append(gt_ellipses[i][j].x_vec)
            GT_para.y_vec.append(gt_ellipses[i][j].y_vec)

    np.save(save_fold + '/GT_para_lambda_maj.npy', GT_para.lambda_maj)
    np.save(save_fold + '/GT_para_x_com.npy', GT_para.x_com)
    np.save(save_fold + '/GT_para_y_com.npy', GT_para.y_com)
    np.save(save_fold + '/GT_para_z_com.npy', GT_para.z_com)
    np.save(save_fold + '/GT_para_fa.npy', GT_para.fa)
    np.save(save_fold + '/GT_para_x_vec.npy', GT_para.x_vec)
    np.save(save_fold + '/GT_para_y_vec.npy', GT_para.y_vec)

    return

def chin_save_nii(savefolder, data_img, data_label, data_bi, NIIname):
    data_img = data_img.data.cpu().numpy() # change from torch to numpy
    data_img = np.float32(data_img)
    data_img = data_img - data_img.min()  # normalisation
    data_img = data_img / data_img.max()  # normalisation
    affine_co = np.eye(4)
    save_img = nib.Nifti1Image(data_img, affine=affine_co)

    data_label = data_label.data.cpu().numpy()
    data_label = np.float32(data_label)
    save_label = nib.Nifti1Image(data_label, affine=affine_co)
    NIIname_label = 'label' + NIIname

    data_bi = data_bi.data.cpu().numpy()
    data_bi = np.float32(data_bi)
    save_bi = nib.Nifti1Image(data_bi, affine=affine_co)
    NIIname_bi = 'bi' + NIIname

    # print(savefolder + '/' + NIIname)
    # if os.path.exists(savefolder):
    #     print("path exists.")
    #     nib.save(save_img, savefolder + '/' + NIIname)
    # else:
    #     os.mkdir(savefolder)
    #     nib.save(save_img, savefolder + '/' + NIIname)

    nib.save(save_img, savefolder + '/' + NIIname)
    nib.save(save_label, savefolder + '/' + NIIname_label)
    nib.save(save_bi, savefolder + '/' + NIIname_bi)

    return

def chin_save_nii_and_ell(savefolder, data_img, data_label, data_bi, gt_para, NIIname):
    data_img = data_img.data.cpu().numpy() # change from torch to numpy
    data_img = np.float32(data_img)
    data_img = data_img - data_img.min()  # normalisation
    data_img = data_img / data_img.max()  # normalisation
    affine_co = np.eye(4)
    save_img = nib.Nifti1Image(data_img, affine=affine_co)

    data_label = data_label.data.cpu().numpy()
    data_label = np.float32(data_label)
    save_label = nib.Nifti1Image(data_label, affine=affine_co)
    NIIname_label = 'label' + NIIname

    data_bi = data_bi.data.cpu().numpy()
    data_bi = np.float32(data_bi)
    save_bi = nib.Nifti1Image(data_bi, affine=affine_co)
    NIIname_bi = 'bi' + NIIname
    # print(savefolder + '/' + NIIname)
    # if os.path.exists(savefolder):
    #     print("path exists.")
    #     nib.save(save_img, savefolder + '/' + NIIname)
    # else:
    #     os.mkdir(savefolder)
    #     nib.save(save_img, savefolder + '/' + NIIname)

    nib.save(save_img, savefolder + '/' + NIIname)
    nib.save(save_label, savefolder + '/' + NIIname_label)
    nib.save(save_bi, savefolder + '/' + NIIname_bi)
    np.save(savefolder + '/gt_x_com.npy', gt_para.x_com)
    np.save(savefolder + '/gt_y_com.npy', gt_para.y_com)
    np.save(savefolder + '/gt_z_com.npy', gt_para.z_com)
    np.save(savefolder + '/gt_lambda_maj.npy', gt_para.lambda_maj)
    np.save(savefolder + '/gt_fa.npy', gt_para.fa)
    np.save(savefolder + '/gt_x_vec.npy', gt_para.x_vec)
    np.save(savefolder + '/gt_y_vec.npy', gt_para.y_vec)
    np.save(savefolder + '/data_img.npy', data_img)
    np.save(savefolder + '/data_bi.npy', data_bi)
    np.save(savefolder + '/data_label.npy', data_label)

    return

## Load data
patch_sz = 36 #24
n_samp = 10 #10
n_sub = 1
new_spacing = 1.0
number_of_epochs = 10 #25

suffix = '/DGX_save1_ep10_p36_s10_v5'
save_fold = pwd + suffix
#save_fold = pwd + '/save' #'/results_para'
#os.mkdir(save_fold)

select_gt_criteria = [1, 2, 2]
#train_root_imgt = '/nfs/project/neurovascular/Dataset/VascuSynth/B01'
train_root_imgt = '/home/chayanin/PycharmProjects/2020_v06_vrienv005_MONAI/Data/Stefano/tmp_VascuSynth/B01'
#train_root_dist = '/nfs/home/ctangwiriyasakul/DGXwork/projects/202011_nov/B01_test_dist'
train_root_dist = '/home/chayanin/PycharmProjects/2020_v06_vrienv005_MONAI/Data/Stefano/tmp_VascuSynth/B01_dist'

train_images = sorted(glob.glob(os.path.join(train_root_imgt, 'B01_VSynth*_im.nii.gz')))
train_labels = sorted(glob.glob(os.path.join(train_root_dist, 'B01_VSynth*_dist.nii.gz')))
train_bilabels = sorted(glob.glob(os.path.join(train_root_imgt, 'B01_VSynth*_gt.nii.gz')))
train_files = [{'image': image_name, 'label': label_name, 'bilabel': bilabel_name} for image_name, label_name, bilabel_name in zip(train_images, train_labels, train_bilabels)]
train_files = train_files[:4]

# val_root = '/home/chayanin/PycharmProjects/2020_v06_vrienv005_MONAI/Data/neurovascular/Junction/Validation'
# val_images = sorted(glob.glob(os.path.join(val_root, 'OP', '*.nii.gz')))
# val_labels = sorted(glob.glob(os.path.join(val_root, 'Labels', 'DistanceMap_*.nii.gz')))
# val_bilabels = sorted(glob.glob(os.path.join(val_root, 'Labels_binary', 'Bi_DistanceMap_*.nii.gz')))
# #val_bilabels = sorted(glob.glob(os.path.join(val_root, 'newLabels', 'SegmentedPerSlice_SABRE_*.nii.gz')))
# val_files = [{'image': image_name, 'label': label_name, 'bilabel': bilabel_name} for image_name, label_name, bilabel_name in zip(val_images, val_labels, val_bilabels)]

train_transforms = Compose([
        LoadNiftid(keys=['image', 'label', 'bilabel']),
        AddChanneld(keys=['image', 'label', 'bilabel']),
        Orientationd(keys=['image', 'label', 'bilabel'], axcodes='RAS'),
        NormalizeIntensityd(keys=['image']),
        #RandGaussianNoised(keys=['image'], prob=0.75, mean=0.0, std=1.75),
        #RandRotate90d(keys=['image', 'label', 'bilabel'], prob=0.5, spatial_axes=[0, 2]),
        CropForegroundd(keys=['image', 'label', 'bilabel'], source_key='image'),
        # RandCropByPosNegLabeld(
        #     keys=['image', 'label', 'bilabel'], label_key='bilabel', size=[patch_sz, patch_sz, patch_sz], pos=1, neg=1, num_samples=n_samp
        # ),
        RandCropByPosNegLabeld(
            keys=['image', 'label', 'bilabel'], label_key='bilabel', size=[patch_sz, patch_sz, patch_sz], pos=1, neg=1,num_samples=n_samp,image_threshold=1
        ),
        # Spacingd(keys=['image', 'label', 'bilabel'], pixdim=(new_spacing, new_spacing, new_spacing), interp_order=(2, 0), mode='nearest'),
        ToTensord(keys=['image', 'label', 'bilabel'])
])

## Define CacheDataset and DataLoader for training and validation
train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=n_sub, shuffle=True, num_workers=0, collate_fn=list_data_collate)
#train_loader = DataLoader(train_ds, batch_size=n_sub, shuffle=False, num_workers=0, collate_fn=list_data_collate)

## End of load data

# TODO Here should be all the part regarding loading/sampling to be fed in the training loop

model_full = TwoStageDetector(
                nnUnet(input_channels=1, base_num_channels=14, num_pool=3, num_classes=1),
                rpn_head=RPNHead(num_classes=2, num_feat=14, num_channels=14, num_anchors=9),
                rcnn_head=BBoxHead_Conv(num_classes=2, num_channels=14, num_feat=14))

#model_full.backbone = model_full.backbone.to(device)
#model_full.rpn_head = model_full.rpn_head.to(device)
#model_full.rcnn_head = model_full.rcnn_head.to(device)

#if torch.cuda.device_count() > 1:
print("Let's use", torch.cuda.device_count(), "GPUs!")
# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
model_full.backbone = nn.DataParallel(model_full.backbone).to(device)
model_full.backbone.module = nn.DataParallel(model_full.backbone.module).to(device)
model_full.rpn_head = nn.DataParallel(model_full.rpn_head).to(device)
model_full.rpn_head.module = nn.DataParallel(model_full.rpn_head.module).to(device)
model_full.rcnn_head = nn.DataParallel(model_full.rcnn_head).to(device)
model_full.rcnn_head.module = nn.DataParallel(model_full.rcnn_head.module).to(device)

# else:
#     model_full.backbone = model_full.backbone.to(device)
#     model_full.rpn_head = model_full.rpn_head.to(device)
#     model_full.rcnn_head = model_full.rcnn_head.to(device)

#optimizer = optim.Adam(model_full.parameters(), lr=1e-4)
optimizer_backbone = optim.Adam(model_full.backbone.parameters(), lr=1e-4)
optimizer_rpn = optim.Adam(model_full.rpn_head.parameters(), lr=1e-4)
optimizer_rcnn = optim.Adam(model_full.rcnn_head.parameters(), lr=1e-4)
#optimiser_full = optim.Adam(model_full.parameters(), lr=1e-3)
#optimiser_mix = optim.Adam(list(model_full.rpn_head.parameters())+list(model_full.rcnn_head.parameters()), lr=1e-4)


epoch_loss_values = list() #chin added
# Training of backbone using distance maps
keep_sum = []
for epoch in range(number_of_epochs):
    print(epoch)
    model_full.backbone.train()
    model_full.rpn_head.eval()
    model_full.rcnn_head.eval()
    if epoch % 5 == 1:
        ## name = '/Users/csudre/Documents/Teaching/AML/ObjectDetection' \
        ##        '/RPN_ODMix_%d.pt' % epoch
        #name = pwd + '/save/RPN_ODMix_%d.pt' % epoch
        name = pwd + suffix + '/RPN_ODMix_%d.pt' % epoch
        torch.save(model_full.state_dict(), name)

    #with tqdm(total=len(train_loader), file=sys.stdout) as pbar:
    start_time = time.time()
    running_loss = 0
    indb = 0
    countt = 0
    for patch_s in train_loader:
        # print(indb)
        # indb = indb+1
        # print("Martin fuck off")
        if select_gt_criteria[0] == 1:

            ## Type-1 no critesia
            batch_images, batch_labels, batch_bilabels = patch_s['image'], patch_s['label'], patch_s['bilabel'] # Chin
            batch_images = batch_images.type(torch.FloatTensor)
            batch_labels = batch_labels.type(torch.FloatTensor)
            batch_bilabels = batch_bilabels.type(torch.FloatTensor)

            model_full.backbone = nn.DataParallel(model_full.backbone)
            model_full.backbone.module = nn.DataParallel(model_full.backbone.module).to(device)
            model_full.rpn_head = nn.DataParallel(model_full.rpn_head)
            model_full.rpn_head.module = nn.DataParallel(model_full.rpn_head.module).to(device)
            model_full.rcnn_head = nn.DataParallel(model_full.rcnn_head)
            model_full.rcnn_head.module = nn.DataParallel(model_full.rcnn_head.module).to(device)

            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            batch_bilabels = batch_bilabels.to(device)

            # if epoch == 0:
            #     save_fold2_pre = save_fold + '/BackBone'
            #     if os.path.exists(save_fold2_pre):
            #         print("save_fold2_pre is already existed.")
            #     else:
            #         os.mkdir(save_fold2_pre)
            #     save_fold2 = save_fold2_pre + '/epoch' + str(epoch) + '_batch' + str(countt)
            #     os.mkdir(save_fold2)
            #
            #     for i in range(batch_images.shape[0]):
            #         # keep_or_not = 0  # 0 means keep
            #         NIIname = 'Data' + str(i) + '_'  + '.nii.gz'
            #         savefolder = save_fold2 + '/raw_data_' + str(i)
            #         if os.path.exists(savefolder):
            #             print("savefolder is already existed.")
            #         else:
            #             os.mkdir(savefolder)
            #         chin_save_nii(savefolder, batch_images[i][0], batch_labels[i][0], batch_bilabels[i][0], NIIname)
            #         del NIIname, savefolder

            losses = model_full.forward_train(Variable(batch_images, requires_grad=True),
                                              dist_seg=batch_labels,
                                              training_stage=['backbone'])

            total_loss = losses['backbone']
            print('BB', total_loss)
            # total_loss.requires_grad = True
            model_full.backbone.zero_grad()
            # gc.collect()
            total_loss.backward()
            optimizer_backbone.step()

            # print(model_full.rpn_head.rpn_conv.weight[0])
            running_loss += float(total_loss.item())
            indb += 1
            # pbar.update(1)
            epoch_loss_values.append(total_loss)  # chin added
            countt = countt + 1

            del batch_images, batch_labels, batch_bilabels
            #del gt_ellipses
        else:
            ## Type-2 with criteria
            batch_images, batch_labels, batch_bilabels = patch_s['image'], patch_s['label'], patch_s['bilabel']  # Chin
            #batch_images = batch_images  # Chin
            #batch_labels = batch_labels  # Chin
            #batch_bilabels = batch_bilabels  # Chin

            #batch_images = batch_images.to(device)
            #batch_labels = batch_labels.to(device)
            #batch_bilabels = batch_bilabels.to(device)

            keep_ik = list()  # chin added
            for ik in range(batch_bilabels.shape[0]):
                tmp = np.squeeze(batch_bilabels[ik, 0, :, :, :])
                temp_label2, numb_lab = label(tmp)
                keep_kkk = 0  # 0 mean keep
                seg_temp2 = [None]
                for kkk in range(1, numb_lab + 1):
                    # print(kkk)
                    seg_temp2[0] = (temp_label2 == kkk)
                    #print(np.sum(seg_temp2[0] == True))
                    if np.sum(seg_temp2[0] == True) < 4:  # number of voxels inside each sampled box
                        keep_kkk = 1
                # if tmp.max() == 1 and keep_kkk == 0 and temp_label2.max() > 0:
                if np.array(seg_temp2[0]).size > 0:
                    if keep_kkk == 0 and temp_label2.max() > 0:  # there is something and every object's size is larger than 1.
                        keep_ik.append(ik)
                    del tmp, temp_label2, seg_temp2

            batch_images_use, batch_labels_use, batch_bilabels_use = chin_select_data(keep_ik, patch_sz, batch_images, batch_labels, batch_bilabels)

            gt_ellipses_tmp = create_gt_from_labels(batch_bilabels_use)

            keep_lambda_maj = list()  # chin added
            for i in range(len(gt_ellipses_tmp)):
                keep_or_not = 0  # 0 means keep
                for j in range(len(gt_ellipses_tmp[i])):
                    tmp_lambda_maj = gt_ellipses_tmp[i][j].lambda_maj
                    if tmp_lambda_maj <= 0:
                        keep_or_not = 1
                if keep_or_not == 0:
                    keep_lambda_maj.append(i)

            if len(keep_lambda_maj) < len(keep_ik):
                batch_images_use2, batch_labels_use2, batch_bilabels_use2 = chin_select_data(keep_lambda_maj, patch_sz, batch_images_use, batch_labels_use, batch_bilabels_use)
                gt_ellipses = create_gt_from_labels(batch_bilabels_use2)
            else:
                gt_ellipses = create_gt_from_labels(batch_bilabels_use)
                batch_images_use2 = batch_images_use
                batch_labels_use2 = batch_labels_use
                batch_bilabels_use2 = batch_bilabels_use

            # check additional gt_ellipses's variables (x_com, y_com, z_com, x_vec, y_vec must be greater than zero and less than shape-1)
            keep_ell_paras = list()  # chin added
            for i in range(len(gt_ellipses)):
                keep_or_not = 0  # 0 means keep
                for j in range(len(gt_ellipses[i])):
                    tmp_lambda_maj = gt_ellipses[i][j].lambda_maj
                    #tmp_para = np.array([gt_ellipses[i][j].x_com, gt_ellipses[i][j].y_com, gt_ellipses[i][j].z_com])
                    tmp_para = np.array([int(gt_ellipses[i][j].x_com), int(gt_ellipses[i][j].y_com), int(gt_ellipses[i][j].z_com)])
                    if tmp_lambda_maj <= 0:
                        keep_or_not = 1
                    else:
                        if np.sum(tmp_para <= 0) > 0 or np.sum(tmp_para >= patch_sz - 2) > 0:  # error
                            #if np.sum(tmp_para <= 0.0) > 0 or np.sum(tmp_para >= patch_sz - 1.6) > 0:  # error
                            # if np.sum(tmp_para == 0.0) > 0 or np.sum(tmp_para >= patch_sz - 1.6) > 0:  # error
                            keep_or_not = 1
                    del tmp_lambda_maj, tmp_para
                if keep_or_not == 0:
                    keep_ell_paras.append(i)

            if len(keep_ell_paras) < len(keep_lambda_maj):
                batch_images_use3, batch_labels_use3, batch_bilabels_use3 = chin_select_data(keep_ell_paras, patch_sz, batch_images_use2, batch_labels_use2, batch_bilabels_use2)
                gt_ellipses2 = create_gt_from_labels(batch_bilabels_use3)
            else:
                gt_ellipses2 = create_gt_from_labels(batch_bilabels_use2)
                batch_images_use3 = batch_images_use2
                batch_labels_use3 = batch_labels_use2
                batch_bilabels_use3 = batch_bilabels_use2

            keep_sum.append(batch_images_use3.sum())
            #print(batch_images_use3.shape)
            #print(len(gt_ellipses2))
            # for i in range(len(gt_ellipses2)):
            #     #keep_or_not = 0  # 0 means keep
            #     for j in range(len(gt_ellipses2[i])):
            #         print("lambda_maj" + str(gt_ellipses2[i][j].lambda_maj))
            #         print("para" + str(np.array([gt_ellipses2[i][j].x_com, gt_ellipses2[i][j].y_com, gt_ellipses2[i][j].z_com])))

            # ###### LOSSS
            # losses = model_full.forward_train(Variable(batch_images_use3, requires_grad=True),
            #                                   dist_seg=batch_labels_use3,
            #                                   training_stage=['backbone'])
            if len(gt_ellipses2) > 0:
                # save_fold2_pre = save_fold + '/BackBone'
                # if os.path.exists(save_fold2_pre):
                #     print("save_fold2_pre is already existed.")
                # else:
                #     os.mkdir(save_fold2_pre)
                # save_fold2 = save_fold2_pre + '/epoch' + str(epoch) + '_batch' + str(countt)
                # os.mkdir(save_fold2)
                # chin_save_Ellipes(save_fold2, gt_ellipses2)
                #
                # if epoch == 0:
                #     for i in range(len(gt_ellipses2)):
                #         # keep_or_not = 0  # 0 means keep
                #         for j in range(len(gt_ellipses2[i])):
                #             NIIname = 'Data' + str(i) + '_' + str(j) + '.nii.gz'
                #             savefolder = save_fold2 + '/raw_data_' + str(i) + '_' + str(j)
                #             if os.path.exists(savefolder):
                #                 print("savefolder is already existed.")
                #             else:
                #                 os.mkdir(savefolder)
                #             chin_save_nii_and_ell(savefolder, batch_images_use3[i][0], batch_labels_use3[i][0], batch_bilabels_use3[i][0], gt_ellipses2[i][j], NIIname)
                #             del NIIname, savefolder


                batch_images_use3 = batch_images_use3.to(device)
                batch_labels_use3 = batch_labels_use3.to(device)
                batch_bilabels_use3 = batch_bilabels_use3.to(device)

                ###### LOSSS

                losses = model_full.forward_train(Variable(batch_images_use3, requires_grad=True),
                                                  dist_seg=batch_labels_use3,
                                                  training_stage=['backbone'])

                total_loss = losses['backbone']
                print('BB', total_loss)
                # total_loss.requires_grad = True
                model_full.backbone.zero_grad()
                # gc.collect()
                total_loss.backward()
                optimizer_backbone.step()

                # print(model_full.rpn_head.rpn_conv.weight[0])
                running_loss += float(total_loss.item())
                indb += 1

                epoch_loss_values.append(total_loss)  # chin added
                countt = countt + 1
            else:
                countt = countt + 1

            del batch_images_use2, batch_labels_use2, batch_bilabels_use2
            del batch_images_use3, batch_labels_use3, batch_bilabels_use3
            del gt_ellipses2, gt_ellipses_tmp
            del batch_images, batch_labels, batch_bilabels
            del gt_ellipses

    end_time = time.time()

keep_sum_NP = np.zeros((len(keep_sum),1))
epoch_loss_values_NP = np.zeros((len(epoch_loss_values),1))
for i in range(len(epoch_loss_values)):
    epoch_loss_values_NP[i] = epoch_loss_values[i].data.cpu().numpy()
    #keep_sum_NP[i] = keep_sum[i].data.cpu().numpy()
np.save(save_fold + '/epoch_loss_values.npy', epoch_loss_values_NP)
#np.save(save_fold + '/keep_sum.npy', keep_sum_NP)
print("chin")