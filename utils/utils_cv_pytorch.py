import torch
import numpy as np
import torch.nn as nn
from config import device

'''
:param Pintput：[[box1_1], [box1_2],...]，Poutput：[[box2_1], [box2_2], ...]
计算Pinput的每一个box和Poutput的box的iou，eg. ious = [[iou1_1_2_1, iou1_1_2_2, ...], [iou1_2_2_1, iou1_2_2_2, ...]]
'''
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



def PNms(out):
    for i in range(1):
        index = torch.eq(out[:, 0], i)
        Ppart = out[index, :]
        for i in range(len(Ppart)):
            parti = Ppart[i]





# ious = PComputeIOU(Pinput=torch.from_numpy(np.array([[1.5, 1.5, 1, 1], [1.5, 1.5, 1, 1]])), Poutput=torch.from_numpy(np.array([[2, 2, 2, 2], [2, 2, 2, 2]])), xyxy=False)
# print(ious)