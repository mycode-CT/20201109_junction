import torch
from torch import nn
import torch.nn.functional as F

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class RPNHead(nn.Module):
    #def __init__(self, num_classes, num_channels, num_feat):
    def __init__(self, num_classes, num_channels, num_feat, num_anchors=1): # Chin
        super(RPNHead, self).__init__()
        self.in_channels = num_channels
        self.feat_channels = num_feat
        self.cls_out_channels = num_classes
        self.num_anchors = 1 # Chin

        self.rpn_conv = nn.Conv3d(self.in_channels, self.feat_channels, 3, padding=1)
        #self.rpn_cls = nn.Conv3d(self.feat_channels, self.num_anchors * self.cls_out_channels, 1)
        self.rpn_cls = nn.Conv3d(self.feat_channels, self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv3d(self.feat_channels,  7, 1)

        initialize_weights(self)

    def forward(self, x):
        first_conv = F.relu(self.rpn_conv(x))
        cls_pred = self.rpn_cls(first_conv)
        bbox_pred = self.rpn_reg(first_conv)

        ### option 2 ### FA is 4th dimension (must be between 0 tp 1), lamda is 3rd (must be > 0)
        m_tmp = nn.Sigmoid()
        bbox_pred[:, 4, ...] = m_tmp(bbox_pred[:, 4, ...])  # fa
        position_neg = torch.where(bbox_pred[:, 3, :, :, :] < 0)  # lambda_maj
        bbox_pred[:, 3, position_neg[1], position_neg[2], position_neg[3]] = torch.abs(bbox_pred[:, 3, position_neg[1], position_neg[2], position_neg[3]])
        ### End of option 2 ###

        return cls_pred, bbox_pred, first_conv

class BBoxHead(nn.Module):
    def __init__(self, num_classes, num_channels, num_feat):
        super(BBoxHead, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_feat = num_feat
        self.fully_conn_1 = nn.Linear(num_channels, num_feat)
        self.fully_conn_2 = nn.Linear(num_feat, num_feat)
        self.fc_reg = nn.Linear(num_feat, 7)
        self.fc_cls = nn.Linear(num_feat, num_classes)
        initialize_weights(self)

    def forward(self, x):
        x = x.flatten(1)
        # print(x.shape)
        fc1 = F.relu(self.fully_conn_1(x))
        fc2 = F.relu(self.fully_conn_2(fc1))
        cls_pred = self.fc_cls(fc2)
        bbox_pred = self.fc_reg(fc2)
        return cls_pred, bbox_pred

class BBoxHead_Conv(nn.Module):
    def __init__(self, num_classes, num_channels, num_feat):
        super(BBoxHead_Conv, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_feat = num_feat
        self.conv1_reg = nn.Conv3d(num_channels, num_feat, 3, padding=1)
        self.conv2_reg = nn.Conv3d(num_feat, num_feat, 3, padding=1)
        self.conv3_reg = nn.Conv3d(num_feat, num_feat, 3, padding=1)
        self.conv4_reg = nn.Conv3d(num_feat, num_feat, 3, padding=1)
        #self.fc_cls = nn.avgpool3d(num_feat)
        self.fc_reg = nn.Linear(num_feat, 7)
        self.conv1_cls = nn.Conv3d(num_channels, num_feat, 3, padding=1)
        self.conv2_cls = nn.Conv3d(num_feat, num_feat, 3, padding=1)
        self.conv3_cls = nn.Conv3d(num_feat, num_feat, 3, padding=1)
        self.conv4_cls = nn.Conv3d(num_feat, num_feat, 3, padding=1)
        #self.fc_cls = nn.avgpool3d(num_feat, num_classes)
        self.fc_cls = nn.Linear(num_feat, num_classes)
        initialize_weights(self)

    def forward(self, x):
        reg_conv1 = F.relu(self.conv1_reg(x))
        reg_conv2 = F.relu(self.conv2_reg(reg_conv1))
        reg_conv3 = F.relu(self.conv3_reg(reg_conv2))
        reg_conv4 = F.relu(self.conv4_reg(reg_conv3))
        reg_avg5 = reg_conv4.mean(-1).mean(-1).mean(-1) # added 2020.08.11
        #final_reg = F.relu(self.fc_reg(reg_conv4))
        final_reg = F.relu(self.fc_reg(reg_avg5))

        cls_conv1 = F.relu(self.conv1_cls(x))
        cls_conv2 = F.relu(self.conv2_cls(cls_conv1))
        cls_conv3 = F.relu(self.conv3_cls(cls_conv2))
        cls_conv4 = F.relu(self.conv4_cls(cls_conv3))
        cls_avg5 = cls_conv4.mean(-1).mean(-1).mean(-1) # added 2020.08.11
        #final_cls = F.relu(self.fc_reg(cls_conv4))
        final_cls = F.relu(self.fc_cls(cls_avg5))

        return final_cls, final_reg