import torch
import torch.nn as nn
from model import *
from utils.utils_cv_pytorch import *

def infer(p, anchors, stride):
    x = torch.sigmoid(p[..., 0])
    y = torch.sigmoid(p[..., 1])
    w = p[..., 2]
    h = p[..., 3]
    pred_conf = torch.sigmoid(p[..., 4])
    pred_cls = torch.sigmoid(p[..., 5:])
    grid_x = torch.arange(p.size(3)).repeat(p.size(2), 1).view([1, 1, p.size(2), p.size(3)]).type(torch.FloatTensor)


    scaled_anchors = torch.FloatTensor([(a_w / stride , a_h / stride) for a_w, a_h in anchors])
    anchor_w = scaled_anchors[:, 0:1].view((1, len(anchors), 1, 1))
    anchor_h = scaled_anchors[:, 1:2].view((1, len(anchors), 1, 1))

    grid_y = torch.arange(p.size(2)).repeat(p.size(3), 1).t().view(1, 1, p.size(2), p.size(3)).type(torch.FloatTensor)
    pred_boxes = torch.FloatTensor(p[..., :4].shape)
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

    mask = pred_conf > 0.8
    if torch.max(mask) == 0:
        return torch.tensor([[-1, 0, 0, 0, 0]], dtype=torch.int32)

    centerx = (pred_boxes[..., 0][mask] * stride).unsqueeze(0).transpose(1, 0).int()
    centery = (pred_boxes[..., 1][mask] * stride).unsqueeze(0).transpose(1, 0).int()
    w = (pred_boxes[..., 2][mask] * stride).unsqueeze(0).transpose(1, 0).int()
    h = (pred_boxes[..., 3][mask] * stride).unsqueeze(0).transpose(1, 0).int()
    cls = torch.argmax(pred_cls[mask], dim=1).unsqueeze(0).transpose(1, 0).int()

    # print("centerx:", cls)
    # print("centerx:", centerx)
    # print("centerx:", centery)
    # print("centerx:", w)
    out = torch.cat((cls, centerx, centery, w, h), 1)
    return out
    # print("out:", out)
    # print("pred:", pred_boxes[..., 3][0, 0, 13, 13])

anchors = [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]
img = cv2.imread("./data/train/000138.jpg", 1)
img, ratio, new_unpad, (dw, dh) = utils.letterbox(img, new_shape=(416, 416))
imgCopy = img.copy()
# cv2.imwrite("./resize.jpg", img)
img = np.array(img, dtype=np.float32)   #通过np.array的图片的shape都会是[h, w, c]
img = np.transpose(img, (2, 0, 1))
img = img[np.newaxis, :] / 255.0

model = torch.load("./yolov3.pkl", map_location = "cpu")
p13, p26, p52 = model(torch.from_numpy(img))
out1 = infer(p13, anchors[6:9], 32)
out2 = infer(p26, anchors[3:6], 16)
out3 = infer(p52, anchors[0:3], 8)
out = torch.cat((out1, out2, out3), 0)
resizeOut = torch.zeros_like(out)
resizeOut[..., 1] = out[..., 1] - out[..., 3] / 2
resizeOut[..., 2] = out[..., 2] - out[..., 4] / 2
resizeOut[..., 3] = out[..., 1] + out[..., 3] / 2
resizeOut[..., 4] = out[..., 2] + out[..., 4] / 2

for i in range(resizeOut.shape[0]):
    imgCopy = cv2.rectangle(imgCopy, (resizeOut[i][1], resizeOut[i][2]), (resizeOut[i][3], resizeOut[i][4]), (255, 0, 0), 2)
# imgCopy = cv2.rectangle(imgCopy, (resizeOut[1][1], resizeOut[1][2]), (resizeOut[1][3], resizeOut[1][4]), (255, 0, 0), 2)
# imgCopy = cv2.rectangle(imgCopy, (resizeOut[2][1], resizeOut[2][2]), (resizeOut[2][3], resizeOut[2][4]), (255, 0, 0), 2)


cv2.imshow("imgCopy", imgCopy)
cv2.waitKey(0)
# PNms(out)