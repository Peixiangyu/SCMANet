import numpy as np
import torch
import torch.nn as nn
from PIL.Image import Image
from function import DEVICE, calc_ss_loss, calc_remd_loss, calc_moment_loss, calc_mse_loss, calc_histogram_loss, \
    mean_variance_norm
from histoloss import RGBuvHistBlock



class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.vgg = vgg19[:44]
        self.vgg.load_state_dict(torch.load('/kaggle/input/adascma1/encoder.pth', map_location='cuda'), strict=False)
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.cwct = cWCT(256)
        self.align1 = SCMA(512)
        self.align2 = SCMA(512)
        self.align3 = SCMA(512)
        self.catunit = CatUnit(512)

        self.decoder = decoder
        self.criterionMSE = torch.nn.MSELoss()

        self.MSE_instance_loss = torch.nn.MSELoss(reduction='none').cuda()  # 不对每个损失进行求和与平均

        self.hist = RGBuvHistBlock(insz=64, h=256,
                                   intensity_scale=True,
                                   method='inverse-quadratic',
                                   device=DEVICE)  # 没看懂

        if args.pretrained == True:
            self.cwct.load_state_dict(torch.load('/kaggle/input/cwct-scma-10w/CWCT.pth', map_location='cuda'),
                                      strict=True)
            self.align1.load_state_dict(torch.load('/kaggle/input/cwct-scma-10w/SCMA1.pth', map_location='cuda'),
                                        strict=True)
            self.align2.load_state_dict(torch.load('/kaggle/input/cwct-scma-10w/SCMA2.pth', map_location='cuda'),
                                        strict=True)
            self.align3.load_state_dict(torch.load('/kaggle/input/cwct-scma-10w/SCMA3.pth', map_location='cuda'),
                                        strict=True)
            self.catunit.load_state_dict(torch.load('/kaggle/input/cwct-scma-10w/CatUnit.pth', map_location='cuda'),
                                         strict=True)
            self.decoder.load_state_dict(torch.load('/kaggle/input/cwct-scma-10w/decoder.pth', map_location='cuda'),
                                         strict=False)

        if args.requires_grad == False:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, Ic, Is):
        feat_c = self.forward_vgg(Ic)
        feat_s = self.forward_vgg(Is)

        Fc2, Fs2 = feat_c[2], feat_s[2]
        Fcs1 = self.cwct(Fc2, Fs2)  # relu_3层{5,256,64,64}

        Fc3, Fs3 = feat_c[3], feat_s[3]
        Fcs2 = self.align1(Fc3, Fs3)
        Fcs3 = self.align2(Fcs2, Fs3)
        Fcs4 = self.align3(Fcs3, Fs3)  # 经三轮PAMA模块融合后的风格化特征

        # 这里写一个特征融合模块，模块权重全部保存下来
        Fcs = self.catunit(Fcs1, Fcs4)
        Ics3 = self.decoder(Fcs)  # 将最终的风格化特征转换为风格化图像

        if self.args.training == True:
            Ics1 = self.decoder(Fcs2)
            Ics2 = self.decoder(Fcs3)
            Irc = self.decoder(Fc3)
            Irs = self.decoder(Fs3)
            feat_cs1 = self.forward_vgg(Ics1)
            feat_cs2 = self.forward_vgg(Ics2)
            feat_cs3 = self.forward_vgg(Ics3)
            feat_rc = self.forward_vgg(Irc)
            feat_rs = self.forward_vgg(Irs)

            content_loss1, remd_loss1, moment_loss1, color_loss1 = 0.0, 0.0, 0.0, 0.0
            content_loss2, remd_loss2, moment_loss2, color_loss2 = 0.0, 0.0, 0.0, 0.0
            content_loss3, remd_loss3, moment_loss3, color_loss3 = 0.0, 0.0, 0.0, 0.0
            loss_rec = 0.0

            for l in range(2, 5):
                content_loss1 += self.args.w_content1 * calc_ss_loss(feat_cs1[l], feat_c[l])
                remd_loss1 += self.args.w_remd1 * calc_remd_loss(feat_cs1[l], feat_s[l])
                moment_loss1 += self.args.w_moment1 * calc_moment_loss(feat_cs1[l], feat_s[l])

                content_loss2 += self.args.w_content2 * calc_ss_loss(feat_cs2[l], feat_c[l])
                remd_loss2 += self.args.w_remd2 * calc_remd_loss(feat_cs2[l], feat_s[l])
                moment_loss2 += self.args.w_moment2 * calc_moment_loss(feat_cs2[l], feat_s[l])

                content_loss3 += self.args.w_content3 * calc_ss_loss(feat_cs3[l], feat_c[l])
                remd_loss3 += self.args.w_remd3 * calc_remd_loss(feat_cs3[l], feat_s[l])
                moment_loss3 += self.args.w_moment3 * calc_moment_loss(feat_cs3[l], feat_s[l])

                loss_rec += 0.5 * calc_mse_loss(feat_rc[l], feat_c[l]) + 0.5 * calc_mse_loss(feat_rs[l], feat_s[l])
            loss_rec += 25 * calc_mse_loss(Irc, Ic)
            loss_rec += 25 * calc_mse_loss(Irs, Is)

            if self.args.color_on:
                color_loss1 += self.args.w_color1 * calc_histogram_loss(Ics1, Is, self.hist)
                color_loss2 += self.args.w_color2 * calc_histogram_loss(Ics2, Is, self.hist)
                color_loss3 += self.args.w_color3 * calc_histogram_loss(Ics3, Is, self.hist)

            loss1 = (content_loss1 + remd_loss1 + moment_loss1 + color_loss1) / (
                        self.args.w_content1 + self.args.w_remd1 + self.args.w_moment1 + self.args.w_color1)
            loss2 = (content_loss2 + remd_loss2 + moment_loss2 + color_loss2) / (
                        self.args.w_content2 + self.args.w_remd2 + self.args.w_moment2 + self.args.w_color2)
            loss3 = (content_loss3 + remd_loss3 + moment_loss3 + color_loss3) / (
                        self.args.w_content3 + self.args.w_remd3 + self.args.w_moment3 + self.args.w_color3)
            loss = loss1 + loss2 + loss3 + loss_rec
            return loss, Ics3  # 返回损失值
        else:
            return Ics3  # 如果训练的参数值不等于Ttrue就返回图像

    def forward_vgg(self, x):
        relu1_1 = self.vgg[:4](x)
        relu2_1 = self.vgg[4:11](relu1_1)
        relu3_1 = self.vgg[11:18](relu2_1)
        relu4_1 = self.vgg[18:31](relu3_1)
        relu5_1 = self.vgg[31:44](relu4_1)
        return [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1]

    def save_ckpts(self):
        torch.save(self.cwct.state_dict(), "/kaggle/working/log/CWCT.pth")
        torch.save(self.align1.state_dict(), "/kaggle/working/log/SCMA1.pth")
        torch.save(self.align2.state_dict(), "/kaggle/working/log/SCMA2.pth")
        torch.save(self.align3.state_dict(), "/kaggle/working/log/SCMA3.pth")
        torch.save(self.catunit.state_dict(), "/kaggle/working/log/CatUnit.pth")
        torch.save(self.decoder.state_dict(), "/kaggle/working/log/decoder.pth")

    # ---------------------------------------------------------------------------------------------------------------


vgg19 = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1,
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

# ---------------------------------------------------------------------------------------------------------------

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),  # relu4_1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),  # relu3_1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),  # relu2_1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

# ---------------------------------------------------------------------------------------------------------------


from collections import OrderedDict


# 定义SKNet与SENet的思想基本上是一致的，只不过使用动态卷积核，并将这些卷积核的通道权重相融合

class SKAttention(nn.Module):

    def __init__(self, channel, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)  # 取最大通道数
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),  # 批量归一化
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)  # 全连接层, self.d指的是输出通道数
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))  # 创建全连接层列表
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w,  将所有的张量按照维度0进行堆叠，形成新的张量

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w 对所有特征图进行求和

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c 平均池化操作
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V

        # cWCT模块


class cWCT(torch.nn.Module):
    def __init__(self, channels, eps=2e-5, use_double=False):  # 进行双精度训练
        super().__init__()
        self.eps = eps
        self.use_double = use_double
        self.conv_ct1 = nn.Conv2d(channels, channels, (3, 3), stride=2)
        self.relu = nn.ReLU()
        self.conv_ct2 = nn.Conv2d(channels, 2 * channels, (3, 3), stride=1)
        self.pad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.pad2 = nn.ReflectionPad2d((1, 1, 1, 1))

    def forward(self, cont_feat, styl_feat):
        cwct_cs = self._transfer(cont_feat, styl_feat)  # 风格迁移模块
        cfs = self.conv_ct1(self.pad1(cwct_cs))
        cfs = self.relu(cfs)
        cfs1 = self.conv_ct2(self.pad2(cfs))
        return cfs1

    def _transfer(self, cont_feat, styl_feat):
        """
        :param cont_feat: [B, N, cH, cW]
        :param styl_feat: [B, N, sH, sW]
        :return color_fea: [B, N, cH, cW]
        """
        B, N, cH, cW = cont_feat.shape
        cont_feat = cont_feat.reshape(B, N, -1)
        styl_feat = styl_feat.reshape(B, N, -1)

        in_dtype = cont_feat.dtype
        if self.use_double:
            cont_feat = cont_feat.double()
            styl_feat = styl_feat.double()

        # whitening and coloring transforms
        whiten_fea = self.whitening(cont_feat)
        color_fea = self.coloring(whiten_fea, styl_feat)

        if self.use_double:
            color_fea = color_fea.to(in_dtype)

        return color_fea.reshape(B, N, cH, cW)

        if self.use_double:
            color_fea = color_fea.to(in_dtype)

        return color_fea.reshape(B, N, cH, cW)

    def cholesky_dec(self, conv, invert=False):
        cholesky = torch.linalg.cholesky if torch.__version__ >= '1.8.0' else torch.cholesky
        try:
            L = cholesky(conv)
        except RuntimeError:
            # print("Warning: Cholesky Decomposition fails")
            iden = torch.eye(conv.shape[-1]).to(conv.device)
            eps = self.eps
            while True:
                try:
                    conv = conv + iden * eps
                    L = cholesky(conv)
                    break
                except RuntimeError:
                    eps = eps + self.eps

        if invert:
            L = torch.inverse(L)

        return L.to(conv.dtype)

    def whitening(self, x):
        mean = torch.mean(x, -1)
        mean = mean.unsqueeze(-1).expand_as(x)
        x = x - mean

        conv = (x @ x.transpose(-1, -2)).div(x.shape[-1] - 1)
        inv_L = self.cholesky_dec(conv, invert=True)

        whiten_x = inv_L @ x

        return whiten_x

    def coloring(self, whiten_xc, xs):
        xs_mean = torch.mean(xs, -1)
        xs = xs - xs_mean.unsqueeze(-1).expand_as(xs)

        conv = (xs @ xs.transpose(-1, -2)).div(xs.shape[-1] - 1)
        Ls = self.cholesky_dec(conv, invert=False)

        coloring_cs = Ls @ whiten_xc
        coloring_cs = coloring_cs + xs_mean.unsqueeze(-1).expand_as(coloring_cs)

        return coloring_cs

    def compute_label_info(self, cont_seg, styl_seg):
        if cont_seg.size is False or styl_seg.size is False:
            return
        max_label = np.max(cont_seg) + 1
        self.label_set = np.unique(cont_seg)
        self.label_indicator = np.zeros(max_label)
        for l in self.label_set:
            # if l==0:
            #   continue
            is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100
            o_cont_mask = np.where(cont_seg.reshape(cont_seg.shape[0] * cont_seg.shape[1]) == l)
            o_styl_mask = np.where(styl_seg.reshape(styl_seg.shape[0] * styl_seg.shape[1]) == l)
            self.label_indicator[l] = is_valid(o_cont_mask[0].size, o_styl_mask[0].size)
        return self.label_set, self.label_indicator

    def resize(self, img, H, W):
        size = (W, H)
        if len(img.shape) == 2:
            return np.array(Image.fromarray(img).resize(size, Image.NEAREST))
        else:
            return np.array(Image.fromarray(img, mode='RGB').resize(size, Image.NEAREST))

    def get_index(self, feat, label):
        mask = np.where(feat.reshape(feat.shape[0] * feat.shape[1]) == label)
        if mask[0].size <= 0:
            return None
        return torch.LongTensor(mask[0])

    # 这里添加一个通道洗牌模块
    def channel_shuffle(self, x, groups):  # 这里groups代表的是组数
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)

        return x

    def interpolation(self):
        # To do
        return


# 注意力模块
class AttentionUnit(nn.Module):
    def __init__(self, channels):
        super(AttentionUnit, self).__init__()
        self.relu6 = nn.ReLU6()
        self.f = nn.Conv2d(channels, channels // 2, (1, 1))
        self.g = nn.Conv2d(channels, channels // 2, (1, 1))
        self.h = nn.Conv2d(channels, channels // 2, (1, 1))
        self.v = nn.Conv2d(channels, channels // 2, (1, 1))

        self.out_conv = nn.Conv2d(channels // 2, channels, (1, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Fc, Fs):
        B, C, H, W = Fc.shape
        f_Fc = self.relu6(self.f(mean_variance_norm(Fc)))
        g_Fs = self.relu6(self.g(mean_variance_norm(Fs)))
        h_Fs = self.relu6(self.h(Fs))
        f_Fc = f_Fc.view(f_Fc.shape[0], f_Fc.shape[1], -1).permute(0, 2, 1)
        g_Fs = g_Fs.view(g_Fs.shape[0], g_Fs.shape[1], -1)

        Attention = self.softmax(torch.bmm(f_Fc, g_Fs))

        h_Fs = h_Fs.view(h_Fs.shape[0], h_Fs.shape[1], -1)

        Fcs = torch.bmm(h_Fs, Attention.permute(0, 2, 1))
        Fcs = Fcs.view(B, C // 2, H, W)
        Fcs = self.relu6(self.out_conv(Fcs))

        return Fcs


# 使用空间注意力模块了，所以需要将这个部分的代码注释掉
# 双插值融合
class FuseUnit(nn.Module):
    def __init__(self, channels):
        super(FuseUnit, self).__init__()

        self.proj1 = nn.Conv2d(2 * channels, channels, (1, 1))
        self.proj2 = nn.Conv2d(channels, channels, (1, 1))
        self.proj3 = nn.Conv2d(channels, channels, (1, 1))

        self.fuse1x = nn.Conv2d(channels, 1, (1, 1), stride=1)
        self.fuse3x = nn.Conv2d(channels, 1, (3, 3), stride=1)
        self.fuse5x_1 = nn.Conv2d(channels, 1, (5, 5), stride=1)

        self.fuse5x_2 = nn.Conv2d(channels, 1, (5, 5), stride=1)
        self.fuse7x = nn.Conv2d(channels, 1, (7, 7), stride=1)
        self.fuse9x = nn.Conv2d(channels, 1, (9, 9), stride=1)

        self.pad3x = nn.ReflectionPad2d((1, 1, 1, 1))
        self.pad5x_1 = nn.ReflectionPad2d((2, 2, 2, 2))
        self.pad5x_2 = nn.ReflectionPad2d((2, 2, 2, 2))
        self.pad7x = nn.ReflectionPad2d((3, 3, 3, 3))
        self.pad9x = nn.ReflectionPad2d((4, 4, 4, 4))
        self.sigmoid = nn.Sigmoid()

    def forward(self, F1, F2):
        Fcat = self.proj1(torch.cat((F1, F2), dim=1))

        F1 = self.proj2(F1)
        F2 = self.proj3(F2)

        # 第一个融合模块
        fusion1 = self.sigmoid(self.fuse1x(Fcat))
        fusion3_1 = self.sigmoid(self.fuse3x(self.pad3x(Fcat)))
        fusion5_1 = self.sigmoid(self.fuse5x_1(self.pad5x_1(Fcat)))
        fusion_1 = (fusion1 + fusion3_1 + fusion5_1) / 3
        fusion_1 = torch.clamp(fusion_1, min=0, max=1.0) * F1 + torch.clamp(1 - fusion_1, min=0, max=1.0) * F2

        fusion5_2 = self.sigmoid(self.fuse5x_2(self.pad5x_2(Fcat)))
        fusion7 = self.sigmoid(self.fuse7x(self.pad7x(Fcat)))
        fusion9 = self.sigmoid(self.fuse9x(self.pad9x(Fcat)))

        fusion_2 = (fusion5_2 + fusion7 + fusion9) / 3
        fusion_2 = torch.clamp(fusion_2, min=0, max=1.0) * F1 + torch.clamp(1 - fusion_2, min=0, max=1.0) * F2

        fusion = fusion_1 + fusion_2

        return fusion


class SCMA(nn.Module):
    def __init__(self, channels):
        super(SCMA, self).__init__()
        self.skattn = SKAttention(channels)
        self.attn = AttentionUnit(channels)
        self.fuse = FuseUnit(channels)
        self.conv_out = nn.Conv2d(channels, channels, (3, 3), stride=1)
        self.conv_in = nn.Conv2d(channels, channels, (3, 3), stride=1)

        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.relu6 = nn.ReLU6()

    def forward(self, Fc, Fs):
        Fc1 = self.skattn(Fc)
        Fs1 = self.skattn(Fs)
        Fcs = self.attn(Fc1, Fs1)
        Fcs = self.relu6(self.conv_out(self.pad(Fcs)))
        Fcs = self.fuse(Fc, Fcs)

        return Fcs


class CatUnit(nn.Module):
    def __init__(self, channels):
        super(CatUnit, self).__init__()
        self.conv_1 = nn.Conv2d(2 * channels, channels, (1, 1), stride=1)

        # 这里写一个前向传播过程

    def forward(self, cs1, cs2):
        Fcs = torch.cat((cs1, cs2), dim=1)  # 将得到的所有风格化特征进行cat融合
        Fcs = self.conv_1(Fcs)

        return Fcs

# ---------------------------------------------------------------------------------------------------------------


