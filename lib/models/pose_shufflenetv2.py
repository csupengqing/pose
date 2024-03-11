from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from einops import rearrange

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x,1,2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x

def depthwise_conv(input_c, output_c, kernel, stride=1, padding=0, bias=False):
    return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel, stride=stride, padding=padding, bias=bias, groups=input_c)

class ShuffleNetV2Block(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ShuffleNetV2Block, self).__init__()
        # channel expansion ratio
        ratio = 5
        self.stride = stride
        assert inplanes % 2 == 0
        branch_features = inplanes // 2
        
        # downsample
        if self.stride == 2:
            # 3x3 1x1
            self.branch1 = nn.Sequential(
                depthwise_conv(inplanes, inplanes, kernel=3, stride=self.stride, padding=1, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.Conv2d(inplanes, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
            # 1x1 3x3 1x1
            self.branch2 = nn.Sequential(
                nn.Conv2d(inplanes, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                depthwise_conv(branch_features, branch_features, kernel=3, stride=self.stride, padding=1,bias=False),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )

        # basic Block
        else:
            # none
            self.branch1 = nn.Sequential()
            # 1x1 3x3 1x1
            self.branch2 = nn.Sequential(
                #1x1
                nn.Conv2d(branch_features, branch_features * ratio, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features * ratio),
                nn.ReLU(inplace=True),
                #depthwise
                nn.Conv2d(branch_features * ratio, branch_features * ratio, kernel_size=3, stride=self.stride, padding=1, bias=False, groups=branch_features),
                nn.BatchNorm2d(branch_features * ratio),
                # 1x1
                nn.Conv2d(branch_features * ratio , branch_features, kernel_size=1,stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
                # nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                # nn.BatchNorm2d(branch_features),
                # nn.ReLU(inplace=True)
            )

    
    def forward(self, x):
        # basic block
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            x1 = self.branch1(x1)
            x2 = self.branch2(x2)
            out = torch.cat((x1,x2), dim=1)
        # downsample
        if self.stride == 2:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            out = torch.cat((x1,x2), dim=1)
        out = channel_shuffle(out, 2)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2_d = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=planes)
        self.conv2_p = nn.Conv2d(planes, planes, kernel_size=1, stride=stride, padding=0 , bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_d(out)
        out = self.conv2_p(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Stem(nn.Module):
    def __init__(self, cfg, in_channels, stem_channels, out_channels, expand_ratio=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        #two 3x3conv
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        #3x3 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1,bias=False),
            # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1,bias=False,groups=in_channels),
            # nn.Conv2d(in_channels, stem_channels, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True)
        )

        #ShuffleBlock
        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        # in = out
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels
        #branch
        self.branch1 = nn.Sequential(
            depthwise_conv(branch_channels,branch_channels, kernel=3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.Conv2d(branch_channels,inc_channels,kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(inc_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(branch_channels,mid_channels,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            depthwise_conv(mid_channels,mid_channels,kernel=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels,branch_channels if stem_channels == self.out_channels else stem_channels,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(branch_channels if stem_channels == self.out_channels else stem_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x1,x2 = x.chunk(2,dim=1)
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        out = torch.cat((x1,x2),dim=1)
        out = channel_shuffle(out,2)
        return out
        # return self.conv(x)

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_inchannels , num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_inchannels, num_channels, stride=1):
        downsample = None
        num_channels = [num_channels[i] // 2 for i in range(len(num_channels))]
        num_inchannels = [num_inchannels[i] // 2 for i in range(len(num_inchannels))]
        if stride != 1 or num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        # layers.append(
        #     block(
        #         num_inchannels[branch_index],
        #         num_channels[branch_index],
        #         stride,
        #         downsample
        #     )
        # )
        num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(0, num_blocks[branch_index]):
            layers.append(
                block(
                    num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_inchannels, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_inchannels, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        num_inchannels = [num_inchannels[i] // 2 for i in range(len(num_inchannels))]
        
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_inchannels[j], 3, 2, 1, bias=False, groups=num_inchannels[j]),
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_inchannels[j], num_inchannels[j], 3, 2, 1, bias=False, groups=num_inchannels[j]),
                                    nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        #split
        x1 = []
        x2 = []
        for i in range(len(x)):
            x_1, x_2 = x[i].chunk(2, dim=1)
            x1.append(x_1)
            x2.append(x_2)
        
        if self.num_branches == 1:
            return [self.branches[0](x1[0])]

        for i in range(self.num_branches):
            x1[i] = self.branches[i](x1[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x1[0] if i == 0 else self.fuse_layers[i][0](x1[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x1[j]
                else:
                    y = y + self.fuse_layers[i][j](x1[j])
            x_fuse.append(self.relu(y))
        # shuffle
        for i in range(len(x_fuse)):
            x_fuse[i] = torch.cat((x_fuse[i],x2[i]),dim=1)
            channel_shuffle(x_fuse[i],2)

        return x_fuse

class PoseHighResolutionNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(PoseHighResolutionNet, self).__init__()
        self.coord_representation = cfg.MODEL.COORD_REPRESENTATION
        assert  cfg.MODEL.COORD_REPRESENTATION in ['simdr', 'sa-simdr', 'heatmap'], 'only simdr and sa-simdr and heatmap supported for pose_resnet_upfree'
        # stem net
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
        #                        bias=False)
        # self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # self.relu = nn.ReLU(inplace=True)
        # self.layer1 = self._make_layer(Bottleneck, 64, 4)

        #stem net
        self.stem = Stem(cfg,3,64,256)

        #stage1
        # downsample = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
        #                            nn.BatchNorm2d(256, momentum=BN_MOMENTUM))
        # self.stage1 = nn.Sequential(Bottleneck(64,64,downsample=downsample))

        #stage2
        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = ShuffleNetV2Block

        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        #stage3
        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = ShuffleNetV2Block

        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        #stage4
        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = ShuffleNetV2Block

        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels)

        #fusion layer
        self.upsample1 = nn.Sequential(
            nn.Conv2d(pre_stage_channels[1], pre_stage_channels[0], 1, 1, 0, bias=False),
            nn.BatchNorm2d(pre_stage_channels[0]),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(pre_stage_channels[2], pre_stage_channels[0], 1, 1, 0, bias=False),
            nn.BatchNorm2d(pre_stage_channels[0]),
            nn.Upsample(scale_factor=4, mode='nearest')
        )
        self.upsample3 = nn.Sequential(
            nn.Conv2d(pre_stage_channels[3], pre_stage_channels[0], 1, 1, 0, bias=False),
            nn.BatchNorm2d(pre_stage_channels[0]),
            nn.Upsample(scale_factor=8, mode='nearest')
        )

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']
        
        # head
        if self.coord_representation == 'simdr' or self.coord_representation == 'sa-simdr':
            self.mlp_head_x = nn.Linear(cfg.MODEL.HEAD_INPUT, int(cfg.MODEL.IMAGE_SIZE[0]*cfg.MODEL.SIMDR_SPLIT_RATIO))
            self.mlp_head_y = nn.Linear(cfg.MODEL.HEAD_INPUT, int(cfg.MODEL.IMAGE_SIZE[1]*cfg.MODEL.SIMDR_SPLIT_RATIO))


    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_pre = len(num_channels_pre_layer)
        num_branches_cur = len(num_channels_cur_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels_pre_layer[i], num_channels_pre_layer[i], 3, 1, 1, bias=False, groups=num_channels_pre_layer[i]),
                            nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 1, 1, 0, bias=False),
                            #nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, inchannels, 3, 2, 1, bias=False, groups=inchannels),
                            nn.Conv2d(inchannels, outchannels, 1, 1, 0, bias=False),
                            # nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        block = ShuffleNetV2Block
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            reset_multi_scale_output = True
            # multi_scale_output is only used last module
            # if not multi_scale_output and i == num_modules - 1:
            #     reset_multi_scale_output = False
            # else:
            #     reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):

        x = self.stem(x)
        # x = self.stage1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list2 = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list2[-1]))
            else:
                x_list.append(y_list2[i])
        y_list3 = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list3[-1]))
            else:
                x_list.append(y_list3[i])
        y_list4 = self.stage4(x_list)

        # fusion
        y_final = y_list4[0] + self.upsample1(y_list4[1]) + self.upsample2(y_list4[2]) + self.upsample3(y_list4[3])

        x_ = self.final_layer(y_final)

        if self.coord_representation == 'heatmap':
            return x_
        elif self.coord_representation == 'simdr' or self.coord_representation == 'sa-simdr':
            x = rearrange(x_, 'b c h w -> b c (h w)')
            pred_x = self.mlp_head_x(x)
            pred_y = self.mlp_head_y(x)
            return pred_x, pred_y

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model