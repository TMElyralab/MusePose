import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

from pose.script.tool import save_videos_from_pil
from pose.script.dwpose import draw_pose



def draw_dwpose(video_path, pose_path, out_path, draw_face):

    # capture video info
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(np.around(fps))
    # fps = get_fps(video_path)
    cap.release()

    # render resolution, short edge = 1024
    k = float(1024) / min(width, height)
    h_render = int(k*height//2 * 2)
    w_render = int(k*width//2 * 2)

    # save resolution, short edge = 768
    k = float(768) / min(width, height)
    h_save = int(k*height//2 * 2)
    w_save = int(k*width//2 * 2)

    poses = np.load(pose_path, allow_pickle=True)
    poses = poses.tolist()

    frames = []
    for pose in tqdm(poses):
        detected_map = draw_pose(pose, h_render, w_render, draw_face)
        detected_map = cv2.resize(detected_map, (w_save, h_save), interpolation=cv2.INTER_AREA)
        # cv2.imshow('', detected_map)
        # cv2.waitKey(0)
        detected_map = cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB)
        detected_map = Image.fromarray(detected_map)
        frames.append(detected_map)
      
    save_videos_from_pil(frames, out_path, fps)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="./UBC_fashion/test", help='dance video dir') 
    parser.add_argument("--pose_dir", type=str, default=None, help='auto makedir')
    parser.add_argument("--save_dir", type=str, default=None, help='auto makedir')
    parser.add_argument("--draw_face", type=bool, default=False, help='whether draw face or not')
    args = parser.parse_args()


    # video dir
    video_dir = args.video_dir

    # pose dir
    if args.pose_dir is None:
        pose_dir = args.video_dir + "_dwpose_keypoints"
    else:
        pose_dir = args.pose_dir

    # save dir 
    if args.save_dir is None:
        if args.draw_face == True:
            save_dir = args.video_dir + "_dwpose"
        else:
            save_dir = args.video_dir + "_dwpose_without_face"
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
    # random.shuffle(video_mp4_paths)
    video_mp4_paths.sort()
    print("Num of videos:", len(video_mp4_paths))


    # draw dwpose
    for i in range(len(video_mp4_paths)):
        video_path = video_mp4_paths[i]
        video_name = os.path.relpath(video_path, video_dir)
        base_name = os.path.splitext(video_name)[0]
        
        pose_path = os.path.join(pose_dir, base_name + '.npy')
        if not os.path.exists(pose_path): 
            print('no keypoint file:', pose_path)

        out_path = os.path.join(save_dir, base_name + '.mp4')
        if os.path.exists(out_path): 
            print('already have rendered pose:', out_path)
            continue

        draw_dwpose(video_path, pose_path, out_path, args.draw_face)
        print(f"Process {i+1}/{len(video_mp4_paths)} video")

    print('all done!')