import os
from os import listdir
from os.path import join
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image, ImageOps
import random
from bicubic import imresize
from Gaussian_downsample import gaussian_downsample
import cv2



def modcrop(img,scale):
    (ih, iw) = img.size
    ih = ih - ( ih % scale)
    iw = iw - ( iw % scale)
    img = img.crop((0,0,ih,iw))
    return img


class DataloadFromFolderTest(data.Dataset): # load test dataset
    def __init__(self, image_dir, scale, scene_name, transform, max_frames=20):
        super(DataloadFromFolderTest, self).__init__()
        base_dir = image_dir if scene_name is None else os.path.join(image_dir, scene_name)
        alist = os.listdir(base_dir)
        alist.sort()
        self.image_filenames = [os.path.join(base_dir, x) for x in alist]
        self.image_filenames = sorted(self.image_filenames)
        self.L = len(alist)
        self.scale = scale
        self.transform = transform # To_tensor
        self.max_frames = max_frames

    def _get_num_frames(self):
        if self.max_frames is None:
            return self.L
        return self.L if self.L <= self.max_frames else self.max_frames

    def get_frame_names(self):
        """Return the base names for the frames that will be loaded."""
        n_frames = self._get_num_frames()
        return [os.path.basename(path) for path in self.image_filenames[:n_frames]]

    def __getitem__(self, index):
        target = []
        nFrames = self._get_num_frames()
        for i in range(nFrames):
            GT_temp = modcrop(Image.open(self.image_filenames[i]).convert('RGB'), self.scale)
            target.append(GT_temp)
        LR = [frame.resize((int(target[0].size[0]/self.scale),int(target[0].size[1]/self.scale)), Image.BICUBIC) for frame in target]
        target = [np.asarray(HR) for HR in target] 
        IN = [np.asarray(IN) for IN in LR]
        target = np.asarray(target)
        IN = np.asarray(IN)
        # if self.scale == 4:
        #     target = np.lib.pad(target, pad_width=((0,0), (2*self.scale,2*self.scale), (2*self.scale,2*self.scale), (0,0)), mode='reflect')
        t, h, w, c = target.shape
        t_lr, h_lr, w_lr, c_lr = IN.shape
        target = target.transpose(1,2,3,0).reshape(h,w,-1) # numpy, [H',W',CT']
        IN = IN.transpose(1,2,3,0).reshape(h_lr, w_lr, -1)
        if self.transform:
            target = self.transform(target) # Tensor, [CT',H',W']
            IN = self.transform(IN)
        target = target.view(c,t,h,w)
        IN = IN.view(c_lr,t_lr,h_lr,w_lr)
        # LR = gaussian_downsample(target, self.scale) # [c,t,h,w]
        # LR = torch.cat((LR[:,1:2,:,:], LR,LR[:,t-1:t,:,:]), dim=1)
        LR = torch.cat((IN[:,1:2,:,:], IN,IN[:,t_lr-1:t_lr,:,:]), dim=1)
        del IN
        return LR, target

    def __len__(self):
        return 1


class DataloadFromVideoTest(data.Dataset):
    """Dataset wrapper that reads frames directly from a video file."""

    def __init__(self, video_path, scale, transform, max_frames=20):
        super().__init__()
        self.video_path = video_path
        self.scale = scale
        self.transform = transform
        self.max_frames = max_frames

        self.frames = self._load_frames()
        self.L = len(self.frames)

    def _get_num_frames(self):
        if self.max_frames is None:
            return self.L
        return self.L if self.L <= self.max_frames else self.max_frames

    def _load_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(modcrop(Image.fromarray(frame), self.scale))
                if self.max_frames is not None and self.max_frames >= 0 and len(frames) >= self.max_frames:
                    break
        finally:
            cap.release()
        if not frames:
            raise ValueError(f"No frames could be read from video: {self.video_path}")
        return frames

    def __getitem__(self, index):
        target = []
        nFrames = self._get_num_frames()
        for i in range(nFrames):
            target.append(self.frames[i])
        LR = [frame.resize((int(target[0].size[0]/self.scale),int(target[0].size[1]/self.scale)), Image.BICUBIC) for frame in target]
        target = [np.asarray(HR) for HR in target]
        IN = [np.asarray(IN) for IN in LR]
        target = np.asarray(target)
        IN = np.asarray(IN)
        t, h, w, c = target.shape
        t_lr, h_lr, w_lr, c_lr = IN.shape
        target = target.transpose(1,2,3,0).reshape(h,w,-1) # numpy, [H',W',CT']
        IN = IN.transpose(1,2,3,0).reshape(h_lr, w_lr, -1)
        if self.transform:
            target = self.transform(target) # Tensor, [CT',H',W']
            IN = self.transform(IN)
        target = target.view(c,t,h,w)
        IN = IN.view(c_lr,t_lr,h_lr,w_lr)
        LR = torch.cat((IN[:,1:2,:,:], IN,IN[:,t_lr-1:t_lr,:,:]), dim=1)
        del IN
        return LR, target

    def __len__(self):
        return 1

