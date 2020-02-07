import torch
import torch.nn as nn
import torch.optim as optim
from loss import *
import utils.utils as utils
import cv2
from utils import dataset
from config import device

class PConvolutional(nn.Module):
    def __init__(self, PinputFiterSize, PoutputFiterSize, PkernelSize, Pstride = 1):
        super(PConvolutional,self).__init__()
        self.Pconv = nn.Conv2d(in_channels=PinputFiterSize, out_channels=PoutputFiterSize, kernel_size = PkernelSize, stride = Pstride, padding=0 if PkernelSize == 1 else 1)
        self.PbatchNorm = nn.BatchNorm2d(PoutputFiterSize)
        self.PleakyRelu = nn.LeakyReLU()

    def forward(self, input):
        return self.PleakyRelu(self.PbatchNorm(self.Pconv(input)))

class PResidualBlock(nn.Module):
    def __init__(self, PinputFiterSize, PoutputFiterSize):
        super(PResidualBlock, self).__init__()
        self.Pconv1 = PConvolutional(PinputFiterSize, int(PoutputFiterSize / 2), PkernelSize=1)
        self.Pconv2 = PConvolutional(int(PoutputFiterSize/2), PoutputFiterSize, PkernelSize=3)
        self.PresConv1 = PConvolutional(PoutputFiterSize, PoutputFiterSize, PkernelSize=1)
        self.PresConv2 = PConvolutional(PoutputFiterSize, PoutputFiterSize, PkernelSize=3)

    def forward(self, input):
        input = self.Pconv1(input)
        input = self.Pconv2(input)
        resOut = self.PresConv1(input)
        resOut = self.PresConv2(resOut)
        return resOut + input



class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.Pconv1 = PConvolutional(PinputFiterSize=3, PoutputFiterSize=32, PkernelSize=3)
        self.Pconv2 = PConvolutional(PinputFiterSize=32, PoutputFiterSize=64, PkernelSize=3, Pstride=2)
        self.PresBlock1 = PResidualBlock(PinputFiterSize=64, PoutputFiterSize=64)
        self.Pconv3 = PConvolutional(PinputFiterSize=64, PoutputFiterSize=128, PkernelSize=3, Pstride=2)
        self.PresBlock2_1 = PResidualBlock(PinputFiterSize=128, PoutputFiterSize=128)
        self.PresBlock2_2 = PResidualBlock(PinputFiterSize=128, PoutputFiterSize=128)
        self.Pconv4 = PConvolutional(PinputFiterSize=128, PoutputFiterSize=256, PkernelSize=3, Pstride=2)
        self.PresBlock3_1 = PResidualBlock(PinputFiterSize=256, PoutputFiterSize=256)
        self.PresBlock3_2 = PResidualBlock(PinputFiterSize=256, PoutputFiterSize=256)
        self.PresBlock3_3 = PResidualBlock(PinputFiterSize=256, PoutputFiterSize=256)
        self.PresBlock3_4 = PResidualBlock(PinputFiterSize=256, PoutputFiterSize=256)
        self.PresBlock3_5 = PResidualBlock(PinputFiterSize=256, PoutputFiterSize=256)
        self.PresBlock3_6 = PResidualBlock(PinputFiterSize=256, PoutputFiterSize=256)
        self.PresBlock3_7 = PResidualBlock(PinputFiterSize=256, PoutputFiterSize=256)
        self.PresBlock3_8 = PResidualBlock(PinputFiterSize=256, PoutputFiterSize=256)
        self.Pconv5 = PConvolutional(PinputFiterSize=256, PoutputFiterSize=512, PkernelSize=3, Pstride=2)
        self.PresBlock4_1 = PResidualBlock(PinputFiterSize=512, PoutputFiterSize=512)
        self.PresBlock4_2 = PResidualBlock(PinputFiterSize=512, PoutputFiterSize=512)
        self.PresBlock4_3 = PResidualBlock(PinputFiterSize=512, PoutputFiterSize=512)
        self.PresBlock4_4 = PResidualBlock(PinputFiterSize=512, PoutputFiterSize=512)
        self.PresBlock4_5 = PResidualBlock(PinputFiterSize=512, PoutputFiterSize=512)
        self.PresBlock4_6 = PResidualBlock(PinputFiterSize=512, PoutputFiterSize=512)
        self.PresBlock4_7 = PResidualBlock(PinputFiterSize=512, PoutputFiterSize=512)
        self.PresBlock4_8 = PResidualBlock(PinputFiterSize=512, PoutputFiterSize=512)
        self.Pconv6 = PConvolutional(PinputFiterSize=512, PoutputFiterSize=1024, PkernelSize=3, Pstride=2)
        self.PresBlock5_1 = PResidualBlock(PinputFiterSize=1024, PoutputFiterSize=1024)
        self.PresBlock5_2 = PResidualBlock(PinputFiterSize=1024, PoutputFiterSize=1024)
        self.PresBlock5_3 = PResidualBlock(PinputFiterSize=1024, PoutputFiterSize=1024)
        self.PresBlock5_4 = PResidualBlock(PinputFiterSize=1024, PoutputFiterSize=1024)

    def forward(self, input):
        input = self.Pconv1(input)
        input = self.Pconv2(input)
        input = self.PresBlock1(input)
        input = self.Pconv3(input)
        input = self.PresBlock2_1(input)
        input = self.PresBlock2_2(input)
        input = self.Pconv4(input)
        input = self.PresBlock3_1(input)
        input = self.PresBlock3_2(input)
        input = self.PresBlock3_3(input)
        input = self.PresBlock3_4(input)
        input = self.PresBlock3_5(input)
        input = self.PresBlock3_6(input)
        input = self.PresBlock3_7(input)
        out52 = self.PresBlock3_8(input)
        input = self.Pconv5(out52)
        input = self.PresBlock4_1(input)
        input = self.PresBlock4_2(input)
        input = self.PresBlock4_3(input)
        input = self.PresBlock4_4(input)
        input = self.PresBlock4_5(input)
        input = self.PresBlock4_6(input)
        input = self.PresBlock4_7(input)
        out26 = self.PresBlock4_8(input)
        input = self.Pconv6(out26)
        input = self.PresBlock5_1(input)
        input = self.PresBlock5_2(input)
        input = self.PresBlock5_3(input)
        out13 = self.PresBlock5_4(input)
        return out13, out26, out52

class PConvolutionalSet(nn.Module):
    def __init__(self, PinputFiterSize, PmiddleFiterSize, PoutputFiterSize):
        super(PConvolutionalSet, self).__init__()
        self.Pconv1_1 = PConvolutional(PinputFiterSize=PinputFiterSize, PoutputFiterSize=PmiddleFiterSize, PkernelSize=1)
        self.Pconv3_1 = PConvolutional(PinputFiterSize=PmiddleFiterSize, PoutputFiterSize=PoutputFiterSize, PkernelSize=3)
        self.Pconv1_2 = PConvolutional(PinputFiterSize=PoutputFiterSize, PoutputFiterSize=PmiddleFiterSize, PkernelSize=1)
        self.Pconv3_2 = PConvolutional(PinputFiterSize=PmiddleFiterSize, PoutputFiterSize=PoutputFiterSize, PkernelSize=3)
        self.Pconv1_3 = PConvolutional(PinputFiterSize=PoutputFiterSize, PoutputFiterSize=PmiddleFiterSize, PkernelSize=1)
        self.Pconv3_3 = PConvolutional(PinputFiterSize=PmiddleFiterSize, PoutputFiterSize=PoutputFiterSize, PkernelSize=3)

    def forward(self, input):
        input = self.Pconv1_1(input)
        input = self.Pconv3_1(input)
        input = self.Pconv1_2(input)
        input = self.Pconv3_2(input)
        input = self.Pconv1_3(input)
        input = self.Pconv3_3(input)
        return input

class YOLOLayer(nn.Module):
    def __init__(self, PclassNum = 80, PanchorNum = 3):
        super(YOLOLayer, self).__init__()
        self.PconvSet13 = PConvolutionalSet(PinputFiterSize=1024, PmiddleFiterSize=512, PoutputFiterSize=1024)
        self.Pconv13 = PConvolutional(PinputFiterSize=1024, PoutputFiterSize=1024, PkernelSize=3)
        self.Pconv13Out = nn.Conv2d(in_channels=1024, out_channels=PanchorNum*(5 + PclassNum), kernel_size=1)

        self.PconvUp13 = PConvolutional(PinputFiterSize=1024, PoutputFiterSize=256, PkernelSize=1)
        self.Pup13   = nn.Upsample(scale_factor=2, mode='nearest')

        self.PconvSet26 = PConvolutionalSet(PinputFiterSize=768, PmiddleFiterSize=256, PoutputFiterSize=512)
        self.Pconv26 = PConvolutional(PinputFiterSize=512, PoutputFiterSize=512, PkernelSize=3)
        self.Pconv26Out = nn.Conv2d(in_channels=512, out_channels=PanchorNum*(5+PclassNum), kernel_size=1)

        self.PconvUp26 = PConvolutional(PinputFiterSize=512, PoutputFiterSize=128, PkernelSize=1)
        self.Pup26   = nn.Upsample(scale_factor=2, mode='nearest')

        self.PconvSet52 = PConvolutionalSet(PinputFiterSize=384, PmiddleFiterSize=128, PoutputFiterSize=256)
        self.Pconv52 = PConvolutional(PinputFiterSize=256, PoutputFiterSize=256, PkernelSize=3)
        self.Pconv52Out = nn.Conv2d(in_channels=256, out_channels=PanchorNum*(5+PclassNum), kernel_size=1)

    def forward(self, out13, out26, out52):
        PoutSet13 = self.PconvSet13(out13)
        Pout13 = self.Pconv13(PoutSet13)
        Pout13 = self.Pconv13Out(Pout13)

        PinSet26 = torch.cat((out26, self.Pup13(self.PconvUp13(PoutSet13))), dim=1)
        PoutSet26 = self.PconvSet26(PinSet26)
        Pout26 = self.Pconv26(PoutSet26)
        Pout26 = self.Pconv26Out(Pout26)

        PinSet52 = torch.cat((out52, self.Pup26(self.PconvUp26(PoutSet26))), dim=1)
        PoutSet52 = self.PconvSet52(PinSet52)
        Pout52 = self.Pconv52(PoutSet52)
        Pout52 = self.Pconv52Out(Pout52)

        return Pout13, Pout26, Pout52

class YOLOV3(nn.Module):
    def __init__(self, PclassNum = 80):
        super(YOLOV3, self).__init__()
        self.model = DarkNet53()
        self.yolo_layer = YOLOLayer(PclassNum)
        self.PclassNum = PclassNum
    def forward(self, input):
        out13, out26, out52 = self.model(input)
        p13, p26, p52 = self.yolo_layer(out13, out26, out52)
        p13 = p13.view(input.shape[0], 3, 5+self.PclassNum, p13.size(2), p13.size(3)).permute(0, 1, 3, 4, 2).contiguous()
        p26 = p26.view(input.shape[0], 3, 5+self.PclassNum, p26.size(2), p26.size(3)).permute(0, 1, 3, 4, 2).contiguous()
        p52 = p52.view(input.shape[0], 3, 5+self.PclassNum, p52.size(2), p52.size(3)).permute(0, 1, 3, 4, 2).contiguous()
        return p13, p26, p52

# img = cv2.imread("./data/train/1.jpg", 1)
# img, ratio, new_unpad, (dw, dh) = utils.letterbox(img, new_shape=(160, 160))
# target = [[[0, 365.5*ratio[0] + dw, 159*ratio[0] + dh, 223*ratio[0], 280*ratio[0]]],
#           [[0, 365.5*ratio[0] + dw, 159*ratio[0] + dh, 223*ratio[0], 280*ratio[0]]]]
#
# print("target:", target)
#
# h, w, c = img.shape
#
#
# target = np.array(target)
# target[:, :, 1] = target[:, :, 1] / w
# target[:, :, 2] = target[:, :, 2] / h
# target[:, :, 3] = target[:, :, 3] / w
# target[:, :, 4] = target[:, :, 4] / h
# print(target)

# target = torch.tensor(target, dtype=torch.float32).to(device)   #label: [class, centerx, centery, w, h]
# anchors = [[66, 84], [66, 84], [66, 84], [66, 84], [66, 84], [66, 84], [66, 84], [66, 84], [66, 84]]
#
# img = np.array(img, dtype=np.float32)   #通过np.array的图片的shape都会是[h, w, c]
# img = np.transpose(img, (2, 0, 1))
# img = img[np.newaxis, :] / 255.0
# img = np.concatenate((img, img), axis=0)


# PbatchSize = 1
# anchors = [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]
#
# data = dataset.dataset("./data/label/", (160, 160))
#
# # model = YOLOV3(PclassNum=1)
# model = torch.load("./yolov3.pkl")
# model = model.to(device)
# # inputs = torch.from_numpy(img).to(device)
# opt_SGD = torch.optim.Adam(model.parameters(),lr=0.0001)
# for i in range(200005):
#     target = []
#     inputs = None
#     for j in range(PbatchSize):
#         inputs2, target2 = data()
#         # print("target2:", target2.shape)
#         inputs2 = inputs2[np.newaxis, :, :, :]
#         if inputs is None:
#             inputs = inputs2
#         else:
#             inputs = np.concatenate((inputs, inputs2), axis=0)
#         target.append(torch.tensor(target2, dtype=torch.float32).to(device))
#
#
#
#     inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
#     opt_SGD.zero_grad()
#     p13, p26, p52 = model(inputs)
#     p = [p52, p26, p13]
#     loss = PLayerLoss(p, target, anchors, Pstrides=[8, 16, 32], PclassNum=1)
#     # loss13 = PLayerLoss(p13, target, anchors[6:9], 32, 1)
#     # loss26 = PLayerLoss(p26, target, anchors[3:6], 16, 1)
#     # loss52 = PLayerLoss(p52, target, anchors[0:3], 8, 1)
#     # loss = loss13 + loss26 + loss52
#     print("loss:", loss)
#     loss.backward()
#     opt_SGD.step()
#     if ((i % 100) == 0) & (i != 0):
#         torch.save(model, "./yolov3.pkl")