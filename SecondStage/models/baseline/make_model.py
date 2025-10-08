import torch
import torch.nn as nn
from .backbones.vit_pytorch import vit_small_patch16_224_FSRA
import torch.nn.functional as F
from .backbones.van import *


class Gem_heat(nn.Module):
    def __init__(self, dim = 768, p=3, eps=1e-6):
        super(Gem_heat, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p)  # initial p
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)


    def gem(self, x, p=3):
        p = F.softmax(p).unsqueeze(-1)
        x = torch.matmul(x,p)
        x = x.view(x.size(0), x.size(1))
        return x

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x,f
            else:
                x = self.classifier(x)
                return x
        else:
            return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)




class build_transformer(nn.Module):
    def __init__(self, opt,num_classes,block = 4 ,return_f=False):
        super(build_transformer, self).__init__()
        self.return_f = return_f

        if opt.backbone == "VIT-S":
            model_path = opt.pretrain_path
            # small
            transformer_name = "vit_small_patch16_224_FSRA"
            self.in_planes = 768

            print('using Transformer_type: {} as a backbone'.format(transformer_name))

            self.transformer = vit_small_patch16_224_FSRA(img_size=(256,256), stride_size=[16, 16], drop_path_rate=0.1,
                                                            drop_rate= 0.0, attn_drop_rate=0.0)
            self.transformer.load_param(model_path)
        elif opt.backbone=="VAN-S":
            #self.transformer = van_small()
            self.transformer = van_large()
            checkpoint = torch.load(opt.pretrain_path)["state_dict"]
            self.transformer.load_state_dict(checkpoint)

        self.num_classes = num_classes

        # self.classifier1 = ClassBlock(768,num_classes,0.5,return_f=return_f)
        self.classifier1 = ClassBlock(self.in_planes,num_classes,0.5,return_f=return_f)
        self.block = block
        for i in range(self.block):
            name = 'classifier_heat' + str(i+1)
            setattr(self, name, ClassBlock(self.in_planes, num_classes, 0.5, return_f=self.return_f))


    def forward(self, x):
        features = self.transformer(x)
        tranformer_feature = self.classifier1(features[:,0])

        if self.block==1:
            return tranformer_feature

        part_features = features[:,1:]

        heat_result = self.get_heartmap_pool(part_features)
        y = self.part_classifier(self.block, heat_result, cls_name='classifier_heat')


        if self.training:
            y = y + [tranformer_feature]
            if self.return_f:
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                return cls, features
        else:
            tranformer_feature = tranformer_feature.view(tranformer_feature.size(0),-1,1)
            y = torch.cat([y,tranformer_feature],dim=2)

        return y




    def get_heartmap_pool(self, part_features, add_global=False, otherbranch=False):
        heatmap = torch.mean(part_features,dim=-1)
        size = part_features.size(1)
        arg = torch.argsort(heatmap, dim=1, descending=True)
        x_sort = [part_features[i, arg[i], :] for i in range(part_features.size(0))]
        x_sort = torch.stack(x_sort, dim=0)

        split_each = size / self.block
        split_list = [int(split_each) for i in range(self.block - 1)]
        split_list.append(size - sum(split_list))
        split_x = x_sort.split(split_list, dim=1)

        split_list = [torch.mean(split, dim=1) for split in split_x]
        part_featuers_ = torch.stack(split_list, dim=2)
        if add_global:
            global_feat = torch.mean(part_features, dim=1).view(part_features.size(0), -1, 1).expand(-1, -1, self.block)
            part_featuers_ = part_featuers_ + global_feat
        if otherbranch:
            otherbranch_ = torch.mean(torch.stack(split_list[1:], dim=2), dim=-1)
            return part_featuers_, otherbranch_
        return part_featuers_

    def part_classifier(self,block, x, cls_name='classifier_lpn'):
        part = {}
        predict = {}
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = cls_name + str(i+1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(block):
            y.append(predict[i])
        if not self.training:
            # return torch.cat(y,dim=1)
            return torch.stack(y, dim=2)
        return y

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))





class RGA_Module(nn.Module):
    """ Region Guided Attention (RGA) Module """

    def __init__(self, in_channels, height=8, width=8):
        super(RGA_Module, self).__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width


        self.fc1 = nn.Linear(in_channels, in_channels // 2, bias=False)
        self.fc2 = nn.Linear(in_channels // 2, in_channels, bias=False)


        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_parts, channels = x.shape


        x_c = x.mean(dim=1)
        x_c = F.relu(self.fc1(x_c))
        x_c = self.sigmoid(self.fc2(x_c))


        x_s = x.view(batch_size, channels, self.height, self.width)
        x_s = F.relu(self.conv1(x_s))
        x_s = self.sigmoid(self.conv2(x_s))
        x_s = x_s.view(batch_size, num_parts, channels)


        x = x * x_c.unsqueeze(1) * x_s
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F



class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



class SPP_Module(nn.Module):
    """ Spatial Pyramid Pooling (SPP) """
    def __init__(self, pool_sizes=[1, 2, 4]):
        super(SPP_Module, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        batch_size, num_parts, channels = x.shape
        pooled_features = []

        for pool_size in self.pool_sizes:
            pooled = F.adaptive_avg_pool1d(x.permute(0, 2, 1), pool_size)  # (B, C, pool_size)
            pooled_features.append(pooled.view(batch_size, channels, -1))

        spp_features = torch.cat(pooled_features, dim=2)
        return spp_features





class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = x.permute(0, 2, 1)  # (B, N, C) -> (B, C, N)
        y = self.gap(x)         # (B, C, N) -> (B, C, 1)
        y = self.conv(y)        # (B, C, 1) -> (B, C, 1)
        y = self.sigmoid(y)
        return (x * y).permute(0, 2, 1)











def make_transformer_model(opt, num_class,block = 4,return_f=False):
    print('===========building transformer===========')
    model = build_transformer(opt, num_class,block=block,return_f=return_f)
    return model
