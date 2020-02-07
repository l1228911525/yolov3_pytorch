import torch
import numpy as np
import utils.utils_cv_pytorch as utils
import torch.nn as nn
import math
from config import device

def PComputeIOU(Pinput, Poutput, xyxy=True):
    if not xyxy:
        x1 = Pinput[...,0] - Pinput[..., 2] / 2
        y1 = Pinput[...,1] - Pinput[..., 3] / 2
        x2 = Pinput[...,0] + Pinput[..., 2] / 2
        y2 = Pinput[...,1] + Pinput[..., 3] / 2
        Pinput[..., 0] = x1
        Pinput[..., 1] = y1
        Pinput[..., 2] = x2
        Pinput[..., 3] = y2

        x1 = Poutput[..., 0] - Poutput[..., 2] / 2
        y1 = Poutput[..., 1] - Poutput[..., 3] / 2
        x2 = Poutput[..., 0] + Poutput[..., 2] / 2
        y2 = Poutput[..., 1] + Poutput[..., 3] / 2
        Poutput[..., 0] = x1
        Poutput[..., 1] = y1
        Poutput[..., 2] = x2
        Poutput[..., 3] = y2

    Pinput = Pinput.float()
    Poutput = Poutput.float()
    ious = torch.zeros(size=(Pinput.size(0), Poutput.size(0)), dtype=torch.float32).to(device)
    for i in range(Pinput.size(0)):
        Pone = Pinput[i]
        Pone = Pone.repeat(Poutput.size(0), 1)
        b1_x1, b1_y1, b1_x2, b1_y2 = Pone[:, 0], Pone[:, 1], Pone[:, 2], Pone[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = Poutput[:, 0], Poutput[:, 1], Poutput[:, 2], Poutput[:, 3]
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)

        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

        b1_area = (b1_x2 - b1_x1)*(b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1)*(b2_y2 - b2_y1)
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16).unsqueeze(0)
        ious[i] = iou
    return ious

def build_target(Ptarget, Panchors, PanchorNum, PclassNum, PgridSize, Pstride, PignoreTh = 0.3):
    nB = len(Ptarget)
    nA = int(PanchorNum / 3)
    nC = PclassNum
    nG = PgridSize
    tx = [torch.zeros(size=(nB, nA, ng, ng), dtype=torch.float32).to(device) for ng in nG]
    ty = [torch.zeros(size=(nB, nA, ng, ng), dtype=torch.float32).to(device) for ng in nG]
    tw = [torch.zeros(size=(nB, nA, ng, ng), dtype=torch.float32).to(device) for ng in nG]
    th = [torch.zeros(size=(nB, nA, ng, ng), dtype=torch.float32).to(device) for ng in nG]
    tconf = [torch.zeros(size=(nB, nA, ng, ng), dtype=torch.float32).to(device) for ng in nG]
    tcls = [torch.zeros(size=(nB, nA, ng, ng, nC), dtype=torch.float32).to(device) for ng in nG]
    clsmask = [0, 0, 0]

    for b in range(nB):
        for t in range(Ptarget[b].shape[0]):
            if Ptarget[b][t].sum() == 0:
                continue
            gx = Ptarget[b][t, 1] * nG[0] * Pstride[0]
            gy = Ptarget[b][t, 2] * nG[0] * Pstride[0]
            gw = Ptarget[b][t, 3] * nG[0] * Pstride[0]
            gh = Ptarget[b][t, 4] * nG[0] * Pstride[0]

            gi = [int(gx / stride) for stride in Pstride]
            gj = [int(gy / stride) for stride in Pstride]

            gt_box = torch.tensor(np.array([0, 0, gw, gh], dtype=np.float32), dtype=torch.float32).unsqueeze(0).to(device)
            anchor_shapes = torch.tensor(np.concatenate((np.zeros((len(Panchors), 2)), np.array(Panchors)), 1), dtype=torch.float32).to(device)
            Piou = PComputeIOU(gt_box, anchor_shapes).squeeze(dim=0)
            Piou[int(torch.argmax(Piou))] = 1
            PanchorMask = (Piou > PignoreTh)
            for i in range(PanchorMask.shape[0]):
                if PanchorMask[i]:
                    tconf[int(i / 3)][b, i%3, gj[int(i / 3)], gi[int(i / 3)]] = 1
                    tx[int(i / 3)][b, i%3, gj[int(i / 3)], gi[int(i / 3)]] = gx / Pstride[int(i / 3)] - gi[int(i / 3)]
                    ty[int(i / 3)][b, i%3, gj[int(i / 3)], gi[int(i / 3)]] = gy / Pstride[int(i / 3)] - gj[int(i / 3)]
                    tw[int(i / 3)][b, i%3, gj[int(i / 3)], gi[int(i / 3)]] = math.log(gw / Panchors[int(i / 3)][0] + 1e-16)
                    th[int(i / 3)][b, i%3, gj[int(i / 3)], gi[int(i / 3)]] = math.log(gh / Panchors[int(i / 3)][1] + 1e-16)
                    PtargetLabel = int(Ptarget[b][t, 0])
                    tcls[int(i / 3)][b, i%3, gj[int(i / 3)], gi[int(i / 3)], PtargetLabel] = 1
                    clsmask[int(i / 3)] = 1
    return tx, ty, tw, th, tconf, tcls, clsmask



def PLayerLoss(p, Ptarget, Panchors, Pstrides, PclassNum):
    PsmoothL1 = nn.SmoothL1Loss(reduction='mean')
    PbceLoss = nn.BCELoss(reduction='mean')
    PceLoss = nn.CrossEntropyLoss()

    x = [torch.sigmoid(px[..., 0]) for px in p]
    y = [torch.sigmoid(py[..., 1]) for py in p]
    w = [pw[..., 2] for pw in p]
    h = [ph[..., 3] for ph in p]

    pred_conf = [torch.sigmoid(pcf[..., 4]) for pcf in p]
    pred_cls = [torch.sigmoid(pcs[..., 5:]) for pcs in p]
    PgridSize = [pp.size(2) for pp in p]

    tx, ty, tw, th, tconf, tcls, clsmask = build_target(Ptarget, Panchors, len(Panchors), PclassNum, PgridSize, Pstrides, 0.3)
    conf_mask_true = [tt.to(torch.bool) for tt in tconf]
    conf_mask_false = [(torch.ones_like(tt, dtype=torch.int) - tt.to(torch.int)).to(torch.bool) for tt in conf_mask_true]
    loss = 0
    for i in range(3):
        if clsmask[i]:
            loss_x = PsmoothL1(x[i][conf_mask_true[i]], tx[i][conf_mask_true[i]])
            # print("loss_x:", loss_x)
            loss_y = PsmoothL1(y[i][conf_mask_true[i]], ty[i][conf_mask_true[i]])
            loss_w = PsmoothL1(w[i][conf_mask_true[i]], tw[i][conf_mask_true[i]])
            loss_h = PsmoothL1(h[i][conf_mask_true[i]], th[i][conf_mask_true[i]])

            tconf = [tt.float() for tt in tconf]

            loss_conf = PbceLoss(pred_conf[i][conf_mask_true[i]], tconf[i][conf_mask_true[i]]) + PbceLoss(
                pred_conf[i][conf_mask_false[i]], tconf[i][conf_mask_false[i]])
            loss_cls = (1 / len(Ptarget)) * PceLoss(pred_cls[i][conf_mask_true[i]], torch.argmax(tcls[i][conf_mask_true[i]], 1))

            loss += (loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls)
        else:
            loss_conf = PbceLoss(pred_conf[i][conf_mask_true[i]], tconf[i][conf_mask_true[i]]) + PbceLoss(
                pred_conf[i][conf_mask_false[i]], tconf[i][conf_mask_false[i]])
            loss += loss_conf
    return loss

# anchors = [[416,416],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]
# target = [[[0, 0.5, 0.5, 0.5, 0.5]]]
# target = torch.tensor(target, dtype=torch.float32).to(device)
# build_target(target, anchors, 9, 80, [52, 26, 13] ,[8, 16, 32])