import math
import os
import random

import cv2
import scipy
import torch
import pandas as pd
from PIL import Image

from scipy import stats
import numpy as np
import glob
import torchvision
from torchvideotransforms import video_transforms, volume_transforms


class Parkinson_former(torch.utils.data.Dataset):
    def __init__(self, args, subset):

        self.subset = subset
        # self.transform = transform

        self.class_idx = args.class_idx  # sport class index(from 0 begin)

        self.args = args
        self.clip_len = args.clip_len
        self.data_root = args.data_root
        self.landmarks_root = args.landmarks_root
        self.split_path = os.path.join(self.data_root, 'PFED5_train.csv')
        self.split_data = pd.read_csv(self.split_path)
        self.split = np.array(self.split_data)
        self.split = self.split[self.split[:, 3] == self.class_idx].tolist()  # stored nums are in str


        if self.subset == 'test':
            self.split_path_test = os.path.join(self.data_root, 'PFED5_test.csv')
            self.split_test = pd.read_csv(self.split_path_test)
            self.split_test = np.array(self.split_test)
            self.split_test = self.split_test[self.split_test[:, 3] == self.class_idx].tolist()


        if self.subset == 'test':
            self.dataset = self.split_test.copy()
        else: # sample 5 clips with different start frame for each video (Augmentation)
            self.dataset = self.split.copy()*5

    def __getitem__(self, index):
        sample_1 = self.dataset[index]
        id_v, label, cls, num_frame = sample_1[0], sample_1[1], int(sample_1[3]), int(sample_1[2])
        patient_id, video_id = id_v.split('_')[-1], id_v[:11]
        id_path = os.path.join(self.data_root, patient_id, video_id)

        data = {}
        data['video'], frame_index = self.load_video(id_path, num_frame)
        if self.args.use_landmark: # only load the features corresponding to the sampled frames
            landmarks_path = os.path.join(self.landmarks_root, patient_id, video_id)
            data['landmark_heatmap'] = self.load_video(landmarks_path, num_frame, frame_index)
        data['final_score'] = label
        # data['video_id'] = id_path
        data['class'] = cls
        return data

    def __len__(self):
        return len(self.dataset)

    def load_short_clips(self, video_frames_list, clip_len, num_frames):
        video_clip = []
        idx = 0
        start_frame = 1
        sample_rate = 1
        frame_index = []
        for i in range(clip_len):
            cur_img_index = start_frame + idx * sample_rate
            # cur_img_path = os.path.join(
            #     video_dir,
            #     "img_" + "{:05}.jpg".format(start_frame + idx * sample_rate))
            #   print(cur_img_path)
            # img = cv2.imread(cur_img_path)
            # video_clip.append(img)
            frame_index.append(cur_img_index)
            if (start_frame + (idx + 1) * sample_rate) > num_frames:
                start_frame = 1
                idx = 0
            else:
                idx += 1
        imgs = [Image.open(video_frames_list[i-1]).convert('RGB') for i in frame_index]
        video_clip.extend(imgs)
        return video_clip, frame_index

    def load_long_clips(self, video_frames_list, clip_len, num_frames):
        video_clip = []

        if self.subset == 'train':
            start_frame = random.randint(1, num_frames - clip_len)
            frame_index = [i for i in range(start_frame, start_frame + clip_len)]
            # print(num_frames, 'index:', frame_index)
            imgs = [Image.open(video_frames_list[i-1]).convert('RGB') for i in
                        range(start_frame, start_frame + clip_len)]
        elif self.subset == 'test':  # sample evenly spaced frames across the sequence for inference
            frame_partition = np.linspace(0, num_frames - 1, num=clip_len, dtype=np.int32)
            frame_index =[i+1 for i in frame_partition]
            # print(num_frames, 'index:', frame_index)
            imgs = [Image.open(video_frames_list[i]).convert('RGB') for i in frame_partition]
        else:
            assert f"subset must be train or test"
        video_clip.extend(imgs)
        return video_clip, frame_index

    def load_video(self, path, num_frame, frame_index=None):
        video_frames_list = sorted((glob.glob(os.path.join(path, '*.jpg'))))
        assert video_frames_list != None, f"check the video dir"
        assert len(
            video_frames_list) == num_frame, f"the number of imgs:{len(video_frames_list)} in {path} must be equal to num_frames:{num_frame}"
        if frame_index is None:
            # if clip length <= input length
            if len(video_frames_list) <= self.clip_len:
                video, frame_index = self.load_short_clips(video_frames_list, self.clip_len, num_frame)
            else:
                video, frame_index = self.load_long_clips(video_frames_list, self.clip_len, num_frame)
            return self.transform(video), frame_index
        else:
            video = []
            imgs = [Image.open(video_frames_list[i-1]).convert('RGB') for i in frame_index]
            video.extend(imgs)
            return self.transform(video, use_landmark=True)
    def transform(self, video, use_landmark=False):
        trans = []
        if use_landmark:
            if self.subset == 'train':
                trans = video_transforms.Compose([
                    video_transforms.RandomHorizontalFlip(),
                    video_transforms.Resize((64, 64)),
                    video_transforms.RandomCrop(56),
                    volume_transforms.ClipToTensor(),
                ])
            elif self.subset == 'test':
                trans = video_transforms.Compose([
                    video_transforms.Resize((64, 64)),
                    video_transforms.CenterCrop(56),
                    volume_transforms.ClipToTensor(),
                ])
        else:
            if self.subset == 'train':
                trans = video_transforms.Compose([
                    video_transforms.RandomHorizontalFlip(),
                    # video_transforms.Resize((256, 256)),
                    video_transforms.RandomCrop(224),
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            elif self.subset == 'test':
                trans = video_transforms.Compose([
                    # video_transforms.Resize((256, 256)),
                    video_transforms.CenterCrop(224),
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        return trans(video)

def train_data_loader(args):
    train_data = Parkinson_former(args,subset='train')
    return train_data

def test_data_loader(args):
    test_data = Parkinson_former(args,subset='test')
    return test_data





