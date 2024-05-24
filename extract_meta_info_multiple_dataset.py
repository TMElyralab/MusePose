import argparse
import json
import os

# python tools/extract_meta_info.py --root_path /path/to/video_dir --dataset_name fashion
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default=[
    "./UBC_fashion/test",
    # "path_of_dataset_1",
    # "path_of_dataset_2",
    ])
parser.add_argument("--save_dir", type=str, default="./meta")
parser.add_argument("--dataset_name", type=str, default="my_dataset")
parser.add_argument("--meta_info_name", type=str, default=None)
parser.add_argument("--draw_face", type=bool, default=False)
args = parser.parse_args()

if args.meta_info_name is None:
    args.meta_info_name = args.dataset_name

# collect all video_folder paths
meta_infos = []
    
for dataset_path in args.root_path:
    video_mp4_paths = set()
    if args.draw_face == True:
        pose_dir = dataset_path + "_dwpose"
    else:
        pose_dir = dataset_path + "_dwpose_without_face"

    for root, dirs, files in os.walk(dataset_path):
        for name in files:
            if name.endswith(".mp4"):
                video_mp4_paths.add(os.path.join(root, name))

    video_mp4_paths = list(video_mp4_paths)
    print(dataset_path)
    print("video num:", len(video_mp4_paths))


    for video_mp4_path in video_mp4_paths:
        relative_video_name = os.path.relpath(video_mp4_path, dataset_path)
        kps_path = os.path.join(pose_dir, relative_video_name)
        meta_infos.append({"video_path": video_mp4_path, "kps_path": kps_path})

save_path = os.path.join(args.save_dir, f"{args.meta_info_name}_meta.json")
json.dump(meta_infos, open(save_path, "w"))
print('data dumped')
print('total pieces of data', len(meta_infos))


import cv2
# check data (cannot read or damaged)
for index, video_meta in enumerate(meta_infos):

    video_path = video_meta[ "video_path"]
    kps_path = video_meta[ "kps_path"]

    video = cv2.VideoCapture(video_path)
    kps = cv2.VideoCapture(kps_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_2 = int(kps.get(cv2.CAP_PROP_FRAME_COUNT))
    assert(frame_count) == (frame_count_2), f"{frame_count} != {frame_count_2} in {video_path}"

    if (index+1) % 100 == 0: print(index+1)

print('data checked, no problem')
