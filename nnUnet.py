import torch.nn as nn
import torch.nn.functional as F
from torch import cat
import os
#from base import BaseModel

class nnUnetConvBlock(nn.Module):
    """
    The basic convolution building block of nnUnet.
    """

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.nonlin = nn.LeakyReLU
        #self.dropout = nn.Dropout3d
        self.dropout = None
        #print('Initialising model without dropout.')
        #self.norm = nn.BatchNorm3d
        self.norm = nn.GroupNorm
        self.conv = nn.Conv3d

        self.nonlin_args = {'negative_slope': 1e-2, 'inplace': True}
        self.dropout_args = {'p': 0.5, 'inplace': True}
        #self.norm_args = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        self.norm_args = {'eps': 1e-5, 'affine': True}
        self.conv_args = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.conv = self.conv(self.input_channels, self.output_channels, **self.conv_args)
        if self.dropout is not None and self.dropout_args['p'] is not None and self.dropout_args['p'] > 0:
            self.dropout = self.dropout(**self.dropout_args)
        else:
            self.dropout = None
        #self.norm = self.norm(num_features=self.output_channels, **self.norm_args)
        #print('Output channels: {} BatchNorm groups: {}'.format(self.output_channels, self.output_channels//4))
        self.norm = self.norm(num_channels=self.output_channels, num_groups=self.output_channels//2, **self.norm_args)
        self.nonlin = self.nonlin(**self.nonlin_args)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.norm(x)
        return self.nonlin(x)

class nnUnetConvBlockStack(nn.Module):
    """
    Concatenates multiple nnUnetConvBlocks.
    """
    def __init__(self, input_channels, output_channels, num_blocks):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stack = nn.Sequential(*([nnUnetConvBlock(input_channels, output_channels)]
                                   +[nnUnetConvBlock(output_channels, output_channels) for _ in range(num_blocks-1)]))


    def forward(self, x):
        return self.stack(x)

class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class nnUnet(nn.Module):
    """
     Stripped-down implementation of nnUnet.

     Notes:
         Based on code in nnUnet repo https://github.com/MIC-DKFZ/nnUNet.
    """
    def __init__(self, input_channels, base_num_channels=30, num_pool=3, num_classes=152):
        super().__init__()
        self.upsample_mode = 'trilinear'
        self.pool = nn.MaxPool3d
        #self.transposed_conv = nn.ConvTranspose3d

        self.downsample_path_convs = []
        self.downsample_path_pooling = []
        self.upsample_path_convs = []
        self.upsample_path_upsampling = []

        # build the downsampling pathway
        # initialise channel numbers for first level
        #input_channels = input_channels # specified as argument
        output_channels = base_num_channels
        for level in range(num_pool):
            # Add two convolution blocks
            self.downsample_path_convs.append(nnUnetConvBlockStack(input_channels, output_channels, 2))

            # Add pooling
            self.downsample_path_pooling.append(self.pool([2,2,2]))

            # Calculate input/output channels for next level
            input_channels = output_channels
            output_channels *= 2

        # now the 'bottleneck'
        final_num_channels = self.downsample_path_convs[-1].output_channels
        self.downsample_path_convs.append(nn.Sequential(nnUnetConvBlockStack(input_channels, output_channels, 1),
                                                        nnUnetConvBlockStack(output_channels, final_num_channels,1)))

        # now build the upsampling pathway
        for level in range(num_pool):
            channels_from_down = final_num_channels
            channels_from_skip = self.downsample_path_convs[-(2 + level)].output_channels
            channels_after_upsampling_and_concat = channels_from_skip * 2

            if level != num_pool-1:
                final_num_channels = self.downsample_path_convs[-(3+level)].output_channels
            else:
                final_num_channels = channels_from_skip

            self.upsample_path_upsampling.append(Upsample(scale_factor=[2,2,2], mode=self.upsample_mode))
            #self.upsample_path_upsampling.append(nn.ConvTranspose3d(channels_from_skip, channels_from_skip, 3, stride=2, output_padding=1))

            # Add two convs
            self.upsample_path_convs.append(nn.Sequential(nnUnetConvBlockStack(channels_after_upsampling_and_concat, channels_from_skip,1),
                                                          nnUnetConvBlockStack(channels_from_skip, final_num_channels,1)))

        # convert to segmentation output
        self.segmentation_output = nn.Conv3d(self.upsample_path_convs[-1][-1].output_channels, num_classes, 1, 1, 0, 1, 1, False)



        # register modules
        self.downsample_path_convs = nn.ModuleList(self.downsample_path_convs)
        self.downsample_path_pooling = nn.ModuleList(self.downsample_path_pooling)
        self.upsample_path_convs = nn.ModuleList(self.upsample_path_convs)
        self.upsample_path_upsampling = nn.ModuleList(self.upsample_path_upsampling)
        self.segmentation_output = nn.ModuleList([self.segmentation_output])

        # run weight initialisation
        from  torch.nn.init  import kaiming_normal_
        for module in self.modules():
             if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
                 kaiming_normal_(module.weight, a=1e-2, nonlinearity='leaky_relu')
                 if module.bias is not None:
                    nn.init.constant_(module.bias,0)


    def forward(self, x):
        skip_connections = []
        #print(x.sum())
        if x.sum() == 0: # I guess I wrote this if condition
            print("NOOOOO")
            os.system("pause")

        for level in range(len(self.downsample_path_convs)-1):
            x = self.downsample_path_convs[level](x)
            skip_connections.append(x)
            #print("Chin Chin")
            #print(x.shape)
            #print(x.sum())
            x = self.downsample_path_pooling[level](x)

        for level in range(len(self.upsample_path_upsampling)):
            x = self.upsample_path_upsampling[level](x)
            # account for differences in spatial dimension due to pooling/upsampling differences. need to look into this more
            diffx= skip_connections[- (1 + level)].shape[2] - x.shape[2]
            diffy= skip_connections[- (1 + level)].shape[3] - x.shape[3]
            diffz= skip_connections[- (1 + level)].shape[4] - x.shape[4]
            x = F.pad(x,[0, diffz,
                         0, diffy,
                         0, diffx])
            x = cat((x,skip_connections[- (1 + level)]), dim=1)
            x = self.upsample_path_convs[level](x)

        # channel = 0
        # z = 45
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(x[0,channel,:,:,z].cpu(), clim=[0,2])
        # plt.clim(x[0,channel,:,:,z].min(),x[0,channel,:,:,z].max())
        # plt.subplot(1,2,2)
        # plt.imshow(skip_connections[- (1 + level)][0,channel,:,:,z].cpu(), clim=[-1,1])
        # plt.clim(skip_connections[- (1 + level)][0,channel,:,:,z].min(), skip_connections[- (1 + level)][0,channel,:,:,z].max())
        # plt.show()
        #
        # import matplotlib.pyplot as plt
        # channel = 0
        # z = 45
        # plt.imshow(x[0, channel, :, :, z].cpu()- skip_connections[- (1 + level)][0,channel,:,:,z].cpu() )
        # plt.show()

        # at the moment no deep supervision so only one operation in segmentation_output
        feat = x
        x = self.segmentation_output[-1](x)
        return feat, x

        # ##chin try something stupid as usual
        # feat = x
        # x_sum = feat.sum(1)
        # return feat, x_sum
        # ## End of chin try something stupid as usual





