import os
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import argparse
import numpy as np
from tqdm import tqdm

from pose.script.dwpose import DWposeDetector
from pose.script.tool import read_frames 




def process_single_video(video_path, detector, root_dir, save_dir):
    # print(video_path)
    video_name = os.path.relpath(video_path, root_dir)
    base_name=os.path.splitext(video_name)[0]
    out_path = os.path.join(save_dir, base_name + '.npy')
    if os.path.exists(out_path): 
        return

    frames = read_frames(video_path)
    keypoints = []
    for frame in tqdm(frames):
        keypoint = detector(frame)
        keypoints.append(keypoint)
      
    result = np.array(keypoints)
    np.save(out_path, result)



def process_batch_videos(video_list, detector, root_dir, save_dir):
    for i, video_path in enumerate(video_list):
        process_single_video(video_path, detector, root_dir, save_dir)
        print(f"Process {i+1}/{len(video_list)} video")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="./UBC_fashion/test")
    parser.add_argument("--save_dir",   type=str, default=None)
    parser.add_argument("--yolox_config",  type=str, default="./pose/config/yolox_l_8xb8-300e_coco.py")
    parser.add_argument("--dwpose_config", type=str, default="./pose/config/dwpose-l_384x288.py")
    parser.add_argument("--yolox_ckpt",  type=str, default="./pretrained_weights/dwpose/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth")
    parser.add_argument("--dwpose_ckpt", type=str, default="./pretrained_weights/dwpose/dw-ll_ucoco_384.pth")
    args = parser.parse_args()

    # make save dir 
    if args.save_dir is None:
        save_dir = args.video_dir + "_dwpose_keypoints"
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # collect all video_folder paths
    video_mp4_paths = set()
    for root, dirs, files in os.walk(args.video_dir):
        for name in files:
            if name.endswith(".mp4"):
                video_mp4_paths.add(os.path.join(root, name))
    video_mp4_paths = list(video_mp4_paths)
    video_mp4_paths.sort()
    print("Num of videos:", len(video_mp4_paths))
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = DWposeDetector(
        det_config = args.yolox_config, 
        det_ckpt = args.yolox_ckpt,
        pose_config = args.dwpose_config, 
        pose_ckpt = args.dwpose_ckpt, 
        keypoints_only=True
        )    
    detector = detector.to(device)
        
    process_batch_videos(video_mp4_paths, detector, args.video_dir, save_dir)
    print('all done!')
