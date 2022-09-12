import cv2
import numpy as np
import random

from pathlib import Path

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

import numpy as np
from matplotlib import pyplot as plt
class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = self.width * self.height * 3 / 2
        self.f = open(filename, 'rb')
        self.shape = (int(self.height*1.5), self.width)

    def read_raw(self):

        raw = self.f.read(int(self.frame_len))
        yuv = np.frombuffer(raw, dtype=np.uint8)
        if yuv.shape[0] == 0:
            return False, None
        yuv = yuv.reshape(self.shape)
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return ret, bgr


if __name__ == "__main__":
    #filename = "data/20171214180916RGB.yuv"
    filename = "Downloads/mother-daughter_cif.yuv"
    size = (288, 352)
    cap = VideoCaptureYUV(filename, size)
    i =0
    frames = []
    while 1:
        ret, frame = cap.read()
        frames.append(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);
        print(rgb)
        print(frame.shape)
#         i+=1
#         print(i)
        if ret:
            image = cv2.imshow("frame", rgb)
            cv2.waitKey(30)
        else:
            break
    print(len(frames))

class YuvVideo(Dataset):
    def __init__(
        self,
        root,
        rnd_interval=False,
        rnd_temp_order=False,
        transform=None,
        split="train",
    ):
        splitdir = Path(root)
        self.samples = [f for f in splitdir.iterdir()]
        self.frames = []
        for filename in self.samples:
            size = (288, 352)
            cap = VideoCaptureYUV(filename, size)
            video = []
            while 1:
                ret, frame = cap.read()
                video.append(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                print(frame.shape)

                if ret:
                    cv2.imshow("frame", rgb)
                    cv2.waitKey(30)
                else:
                    break
            self.frames.append(video)

        self.max_frames = 3  # hard coding for now
        self.rnd_interval = rnd_interval
        self.rnd_temp_order = rnd_temp_order
        self.transform = transform

    def __getitem__(self, index):
        item = [self.transform(self.frames[index + j]) for j in range(3)]

        return item

    def __len__(self):
        len = 0
        for i in range(len(self.frames)):
            len += len(self.frames[i])-2
        return len