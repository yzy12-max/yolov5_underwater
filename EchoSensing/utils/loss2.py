# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
# add
import torch.nn.functional as F


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):   # bce的话可以训练在一个格点存在多个类，ce一个格点就就一个类
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    """
       torch.exp()函数就是求e的多少次方  输入tensor每一个元素经过计算之后返回对应的tensor
       根据下式 对于正常的较大概率的样本 dx对应值为绝对值较小一个负数 假设为-0.12，则-1为-1.12除0.05 为-22.4，
       -22.4 指数化之后为一个很小很小的正数，1-该正数之后得到的值较大 再在loss中乘上之后影响微乎其微
       而对于missing的样本 dx对应为一个稍大的正数 如0.3 减去1之后为-0.7 除以0.05 为 -14
       -14相比-22.4值为指数级增大，因此对应的alpha_factor相比正常样本显著减小 在loss中较小考虑
   """
    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)  # 二值交叉商
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

# QFL:将IOU作为分类的监督信号
class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

# QFL:将IOU作为分类的监督信号
class QFocalLossT(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true): # pre:[]
        assert len(true) == 2, """target for QFL must be a tuple of two elements,
                including category label and quality label, respectively"""
        label, score = true
        pred_sigmoid = pred.sigmoid()  # [177,4] 分数预测
        scale_factor = pred_sigmoid
        # 计算负样本
        zerolabel = scale_factor.new_zeros(pred.shape)  # [N,c]
        loss = self.loss_fcn(pred, zerolabel)* scale_factor.pow(self.gamma)
        # 计算正样本，YOLOV5中没有背景样本，设置了置信度损失
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = pred.shape[1]
        # pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)  # 正样本
        #pos = label>0
        pos = (label > 0).nonzero().squeeze(1)
        i,j = pos[:,0],pos[:,1]
        #pos_label = label[].Long()  # [pos,]
        # positives are supervised by bbox quality (IoU) score，正样本的
        scale_factor = score[j] - pred_sigmoid[i,j]
        loss[i,j] = F.binary_cross_entropy_with_logits(
            pred[i,j], score[j],
            reduction='none') * scale_factor.abs().pow(self.gamma)
        n = pred.shape[0]*pred.shape[1]
        loss = loss.sum(dim=1, keepdim=False)
        loss = loss.sum(dim=0)/n
        return loss

# QFL:将IOU作为分类的监督信号
class VFocalLossT(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true): # pre:[]
        assert len(true) == 2, """target for QFL must be a tuple of two elements,
                including category label and quality label, respectively"""
        label, score = true
        pred_sigmoid = pred.sigmoid()  # [177,4] 分数预测
        scale_factor = pred_sigmoid* (1-self.alpha)
        # 计算负样本
        zerolabel = scale_factor.new_zeros(pred.shape)  # [N,c]
        loss = self.loss_fcn(pred, zerolabel)* scale_factor.pow(self.gamma)
        # 计算正样本，YOLOV5中没有背景样本，设置了置信度损失
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = pred.shape[1]
        # pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)  # 正样本
        #pos = label>0
        pos = (label > 0).nonzero().squeeze(1)
        i,j = pos[:,0],pos[:,1]
        #pos_label = label[].Long()  # [pos,]
        # positives are supervised by bbox quality (IoU) score，正样本的
        scale_factor = score[j]
        loss[i,j] = F.binary_cross_entropy_with_logits(
            pred[i,j], score[j],
            reduction='none') * scale_factor
        n = pred.shape[0]*pred.shape[1]
        loss = loss.sum(dim=1, keepdim=False)
        loss = loss.sum(dim=0)/n
        return loss

class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        # add by my self
        # BCEcls = BCEBlurWithLogitsLoss()
        # BCEobj = BCEBlurWithLogitsLoss()

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            #BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            BCEcls, BCEobj = VFocalLossT(BCEcls, g), BCEobj  # VFocalLossT

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        # [class,(tx,ty,tw,th),(img_index,anchor_index,grid_indices),anchor_type],一个特许证层一个列表
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses pi:[B,NUM_ANCHOR,H,W,C(5+classes)]
        for i, pi in enumerate(p):  # layer index, layer predictions
            # b:一个BATCH里的索引[NUM_P,BATCH]
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n: # pi:[batch,anchor_nums,feat_h,feat_w,class+5]->[n,class+5]:在对应的位置上取出相应的预测数值,
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                # PXY:[NUM_P,2]
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression 预测的是有归一化的偏移值,这里进行了sigmoid
                pxy = pxy.sigmoid() * 2 - 0.5   # 公式,没有+左上角坐标
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]  # 由偏移值转换为了对应特征值的做标志
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                # iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True, alpha=3)  # add by myself
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio,# 将IOU 作为

                # Classification,self.cn=0,self.cp=1
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp  # # 筛选到的正样本对应位置值是cp,[NUM_P,CLASSES],相当于ONE-HOT
                    #lcls += self.BCEcls(pcls, t)  # BCE
                    # 把分类LOSS 改成 QFLOSS
                    target = (t,iou)
                    lcls += self.BCEcls(pcls, target)
                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)   # 置信度LOSS
            # balance用来设置三个feature map对应输出的置信度损失系数(平衡三个feature map的置信度损失)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']  # 乘以权重
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        '''
        P:predict->[检测层数,ANCHOR个数,gridspace长,gridspace宽,类别数+5(xywh+obj)],
        target:[num_target,image_index+class+xywh]
        '''
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], [] # 用来存放结果的
        gain = torch.ones(7, device=self.device)  # normalized to gridspace(就是输出的检测特征层) ,保存放缩到特征图大小的因子
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt),
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices,复制3分对应3个ANCHOR
                                                            # [num_anchor,num_target,image_idx+class+xywh+anchor_idx]
        g = 0.5  # bias,偏移值
        off = torch.tensor(  # 分别对应中心点、左、上、右、下
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl): # 遍历每个检测层
            anchors = self.anchors[i]   # 获取第I层的ANCHOR
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain,放缩
            # # gain = [1, 1, 特征图w, 特征图_h, 特征图w, 特征图_h]
            # Match targets to anchors
            t = targets * gain  # shape(3,n,7) # targets为归一化的,乘以当前层尺度变为当前层的大小
            if nt:  # 如果检测层上有目标
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare,.max(2)[0]返回宽比 高比两者中较大的一个值
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter,[num_Positive_sample,7],初步选择好了那些anchors对应正样本

                # Offsets
                gxy = t[:, 2:4]  # grid xy, # 正样本的xy中心坐标
                gxi = gain[[2, 3]] - gxy  # inverse  #  特征图的长宽-targets的中心
                j, k = ((gxy % 1 < g) & (gxy > 1)).T  # <0.5向下取,左,上
                l, m = ((gxi % 1 < g) & (gxi > 1)).T    # 相减后<0.5向下取,右,下
                j = torch.stack((torch.ones_like(j), j, k, l, m))  # 中心加左,上,右,下,五个
                t = t.repeat((5, 1, 1))[j]   # 复制五分,取出
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()    # 对应多个ANCHOR对应一个GT,.long()表示以网格左上角坐标计算偏移值
            gi, gj = gij.T  # grid indices,把坐标分离开,计算正样本偏移值的左上角坐标

            # Append
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box,.long()表示以网格左上角坐标计算偏移值,范围为(0-1)
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            # [class,(tx,ty,tw,th),(img_index,anchor_index,grid_indices),anchor_type]
        return tcls, tbox, indices, anch
