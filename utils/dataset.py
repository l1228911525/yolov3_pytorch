import glob
import os
import random
import cv2
from utils import utils
# import utils
import numpy as np

class dataset:
    def __init__(self, PlabelPath, PimgSize = (416, 416)):
        self.PtrainLabel = glob.glob(os.path.join(PlabelPath, "*"))
        self.PtrainPic = [x.replace("label", "train").replace("txt", "jpg") for x in self.PtrainLabel]
        self.PimgSize = PimgSize
        # print("pic", self.PtrainPic)
        # print("label:", self.PtrainLabel)

    def __call__(self):
        index = random.randint(0, len(self.PtrainPic) - 1)
        img = cv2.imread(self.PtrainPic[index], 1)
        img, ratio, new_unpad, (dw, dh) = utils.letterbox(img, new_shape=self.PimgSize)

        # imgCopy = img

        img = np.array(img, dtype=np.float32)  # 通过np.array的图片的shape都会是[h, w, c]
        img = np.transpose(img, (2, 0, 1))
        img = img / 255.0

        f = open(self.PtrainLabel[index], 'r')
        Plabel = f.read().split("\n")
        Plabel = [x for x in Plabel if len(x) != 0]
        Plabel = [x.split(",") for x in Plabel]
        Plabel = np.array(Plabel)
        Plabel = Plabel.astype(np.float64)

        Plabel[..., 1] = Plabel[..., 1] * ratio[0] + dw
        Plabel[..., 2] = Plabel[..., 2] * ratio[0] + dh
        Plabel[..., 3] = Plabel[..., 3] * ratio[0] + dw
        Plabel[..., 4] = Plabel[..., 4] * ratio[0] + dh

        # imgCopy = cv2.rectangle(imgCopy, (int(Plabel[0, 1]), int(Plabel[0, 2])), (int(Plabel[0, 3]), int(Plabel[0, 4])), color=(0, 0, 0))

        # cv2.imshow("imgCopy", imgCopy)
        # cv2.waitKey(0)
        PlabelCopy = Plabel.copy()
        PlabelCopy[..., 1] = (Plabel[..., 3] - Plabel[..., 1])/2 + Plabel[..., 1]
        PlabelCopy[..., 2] = (Plabel[..., 4] - Plabel[..., 2])/2 + Plabel[..., 2]
        PlabelCopy[..., 3] = Plabel[..., 3] - Plabel[..., 1]
        PlabelCopy[..., 4] = Plabel[..., 4] - Plabel[..., 2]
        Plabel = PlabelCopy

        Plabel[..., 1] = Plabel[..., 1] / self.PimgSize[0]
        Plabel[..., 2] = Plabel[..., 2] / self.PimgSize[1]
        Plabel[..., 3] = Plabel[..., 3] / self.PimgSize[0]
        Plabel[..., 4] = Plabel[..., 4] / self.PimgSize[1]

        return img, Plabel

# for i in range(10):
#     data = dataset("../data/label")
#     img, Plabel = data()
#     print("Plabel:", Plabel)