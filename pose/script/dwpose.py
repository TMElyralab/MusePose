# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cv2
import torch
import numpy as np
from PIL import Image


import pose.script.util as util

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def draw_pose(pose, H, W, draw_face):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']

    # only the most significant person
    faces = pose['faces'][:1]
    hands = pose['hands'][:2]
    candidate = bodies['candidate'][:18]
    subset = bodies['subset'][:1]

    # draw
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    canvas = util.draw_handpose(canvas, hands)
    if draw_face == True:
        canvas = util.draw_facepose(canvas, faces)

    return canvas

class DWposeDetector:
    def __init__(self, det_config=None, det_ckpt=None, pose_config=None, pose_ckpt=None, device="cpu", keypoints_only=False):
        from pose.script.wholebody import Wholebody

        self.pose_estimation = Wholebody(det_config, det_ckpt, pose_config, pose_ckpt, device)
        self.keypoints_only = keypoints_only
    def to(self, device):
        self.pose_estimation.to(device)
        return self
    '''
        detect_resolution: 短边resize到多少 这是 draw pose 时的原始渲染分辨率。建议1024
        image_resolution: 短边resize到多少 这是 save pose 时的文件分辨率。建议768

        实际检测分辨率：
        yolox: (640, 640)
        dwpose:(288, 384)
    '''

    def __call__(self, input_image, detect_resolution=1024, image_resolution=768, output_type="pil", **kwargs):
        
        input_image = cv2.cvtColor(np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        # cv2.imshow('', input_image)
        # cv2.waitKey(0)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape 
        
        with torch.no_grad():
            candidate, subset = self.pose_estimation(input_image)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)
            
            if self.keypoints_only==True:
                return pose     
            else:   
                detected_map = draw_pose(pose, H, W, draw_face=False)
                detected_map = HWC3(detected_map)
                img = resize_image(input_image, image_resolution)
                H, W, C = img.shape
                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                # cv2.imshow('detected_map',detected_map)
                # cv2.waitKey(0)

                if output_type == "pil":
                    detected_map = cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB)
                    detected_map = Image.fromarray(detected_map)
                    
                return detected_map, pose

