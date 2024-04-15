# -*- coding: utf-8 -*-
import datetime
import glob
import os
import sys
import urllib.request
import urllib.error
import time

import insightface
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import albumentations as A
import cv2
import dlib
import numpy as np
from PIL.Image import Image
from tqdm import tqdm

from opencv_zoo.models.face_detection_yunet import detect
import numpy as np

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

type = sys.getfilesystemencoding()
sys.stdout = Logger('relative_distance_over_first frame.txt')
app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
app.prepare(ctx_id=0, det_size=(640, 640))

transform = A.Compose([A.Resize(256, 256)
    ], keypoint_params=A.KeypointParams(format='xy'))

# ids = ['042-ll042',  '049-bm049',  '066-mg066 ', '096-bg096',  '106-nm106',
#       ' 115-jy115 ', '124-dn124', '043-jh043', ' 052-dr052',  '080-bn080',
#       ' 097-gf097' , '107-hs107 ', '120-kz120', '047-jl047',  '059-fn059 ',
#       ' 092-ch092',  '101-mg101 ', '108-th108',  '121-vw121','048-aa048 ' ,
#       ' 064-ak064 ', '095-tv095',  '103-jk103',  '109-ib109',  '123-jh123'
# ]
ids = [ '095-tv095'
]

root = r"/path/dataset/PAIN/Images"
crop_root = r"/path/dataset/PAIN/Images_crop"
visual_root = r"/path/dataset/PAIN/Images_landmarks_clear"
feature_root = r"/path/dataset/PAIN/Images_landmarks_feature"
color = (0 ,255, 0)

for id in tqdm(ids):
    summary_info = []
    vid_folder = os.path.join(root, id)
    for dirpath, dirnames, filenames in os.walk(vid_folder):
        for vid in tqdm(dirnames):
            vid_path = os.path.join(dirpath, vid)
            img_list = glob.glob(os.path.join(vid_path, '*.png'))
            crop_out_dir = os.path.join(crop_root, id, vid)
            visual_out_dir = os.path.join(visual_root, id, vid)
            feature_save_file = '{}-{}.npy'.format(vid, id)
            feature_out_dir = os.path.join(feature_root, id)

            full_feature = []
            if not os.path.exists(feature_out_dir):
                os.makedirs(feature_out_dir)
            if not os.path.exists(crop_out_dir):
                os.makedirs(crop_out_dir)
            if not os.path.exists(visual_out_dir):
                os.makedirs(visual_out_dir)

            for img_path in sorted(img_list):
                visual_save_path = os.path.join(visual_out_dir, img_path.split('/')[-1])
                crop_save_path = os.path.join(crop_out_dir, img_path.split('/')[-1])
                sub_feature = []

                # #insightface
                img = cv2.imread(img_path)
                faces = app.get(img)
                if faces is not None:
                    for i, face in enumerate(faces):
                        if i >0:
                            continue
                        lmk = face.landmark_2d_106
                        fb = np.round(face.bbox).astype(np.int32)
                        top, left, bottom, right = fb[0],fb[1],fb[2],fb[3]
                        if top < 0 and left >= 0:
                            img_crop, top = img[left:right, 0:bottom], 0
                        elif top >= 0 and left < 0:
                            img_crop, left = img[0:right, top:bottom], 0
                        elif top < 0 and left < 0:
                            img_crop, top, left = img[0:right, 0:bottom], 0 , 0
                        else:
                            img_crop = img[left:right, top:bottom]

                        lmk = np.round(lmk).astype(np.int32)

                        new_lmk = [] # landmarks on cropped face
                        for i in range(33, lmk.shape[0]):  # 106 landmarks remove the contour part
                            new_coor = np.round((lmk[i] - [top, left]))
                            if new_coor[0] < 0:
                                new_coor[0] = 0
                            if new_coor[1] < 0:
                                new_coor[1] = 0
                            if new_coor[0] >= img_crop.shape[1]:
                                new_coor[0] = img_crop.shape[1] - 1
                            if new_coor[1] >= img_crop.shape[0]:
                                new_coor[1] = img_crop.shape[0] - 1
                            new_lmk.append(tuple(new_coor))
                        try:
                            transformed = transform(image=img_crop, keypoints=new_lmk) # resize to (256,256)
                            transformed_image = transformed['image']
                            transformed_keypoints = transformed['keypoints']
                            transformed_keypoints = np.round(transformed_keypoints).astype(np.int32)
                        except:
                            print(img_path,fb)
                            continue
                        img_crop = cv2.resize(img_crop, (256, 256))
                        
                        for i in range(0, len(new_lmk)):  # 73 landmarks
                            p = tuple(transformed_keypoints[i])
                            cv2.circle(transformed_image, p, 1, color, 2)
                            
                    # Convert the facial landmarks into a feature vector
                            sub_feature.append(np.around(np.array([transformed_keypoints[i][0], transformed_keypoints[i][1]]), 2))
                # feature_vector = np.array([shape.part(i).x, shape.part(i).y for i in range(68)])
                #     if len(faces)==0 and len(full_feature)>0:
                        # sub_feature = full_feature[-1]
                    cv2.imwrite(visual_save_path, transformed_image)
                    cv2.imwrite(crop_save_path, img_crop)
                full_feature.append(sub_feature)
            try:
                np.save(os.path.join(feature_out_dir,feature_save_file),
                     full_feature)
            except:
                print(feature_save_file, len(full_feature), full_feature)

            summary_info.append([vid, id, len(img_list)])

    print(' {} done'.format(id))
# compute the distance between centroid and facial landmarks
print('done')