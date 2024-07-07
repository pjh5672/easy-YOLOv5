import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.general import bbox_iou


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441"""
    return 1.0 - 0.5 * eps, 0.5 * eps


class YOLOv5Loss:
    def __init__(self, model, device, imgsz=640):
        m = model.to(device).head
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.cls_pw = 1.0
        self.obj_pw = 1.0
        self.anchor_t = 4.0
        self.balance = [4.0, 1.0, 0.4]
        self.lamb_box = 0.05 * (3 / self.nl) # box loss gain
        self.lamb_cls = 0.5 * (self.nc / 80 * 3 / self.nl) # cls loss gain
        self.lamb_obj = 1.0 * ((imgsz / 640) ** 2 * 3 / self.nl) # obj loss gain (scale with pixels)

        self.anchors = m.anchors
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.cls_pw], device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.obj_pw], device=device))
        self.device= device

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        label_smoothing = 0.0
        self.cp, self.cn = smooth_BCE(eps=label_smoothing)  # positive, negative BCE targets

    def __call__(self, preds, targets):
        lbox = torch.zeros(1, device=self.device)  # box loss
        lcls = torch.zeros(1, device=self.device)  # class loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(preds, targets)  # targets

        # Losses
        for i, pi in enumerate(preds):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj
            
            n = b.shape[0]  # number of targets
            if n:
                # pi[b, a, gj, gi] = (n, nc + 5)
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss

        lbox *= self.lamb_box
        lcls *= self.lamb_cls
        lobj *= self.lamb_obj
        bs = tobj.shape[0]  # batch size

        return (lbox + lcls + lobj) * bs, torch.cat((lbox, lcls, lobj)).detach()


    def build_targets(self, preds, targets):
        na, nt = self.na, targets.shape[0] # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices (3,nt,7)
        
        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], preds[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.anchor_t  # compare
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch