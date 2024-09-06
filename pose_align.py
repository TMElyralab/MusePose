import numpy as np
import argparse
import torch
import copy
import cv2
import os
import moviepy.video.io.ImageSequenceClip
from pose.script.dwpose import DWposeDetector, draw_pose
from pose.script.util import size_calculate, warpAffine_kps

def align_img(img, pose_ori, scales, detect_resolution, image_resolution):
    body_pose = copy.deepcopy(pose_ori['bodies']['candidate'])
    hands, faces = copy.deepcopy(pose_ori['hands']), copy.deepcopy(pose_ori['faces'])

    H_in, W_in = img.shape[:2]
    video_ratio = W_in / H_in
    for keypoints in [body_pose, hands, faces]:
        keypoints[..., 0] *= video_ratio

    # Handle scales
    scale_list = [scales.get(f"scale_{part}", 1.0) for part in 
                  ["neck", "face", "shoulder", "arm_upper", "arm_lower", "hand", "body_len", "leg_upper", "leg_lower"]]
    avg_scale = np.nanmean([s for s in scale_list if not np.isinf(s)])
    scale_list = [s if not np.isinf(s) else avg_scale for s in scale_list]

    # Precompute offsets for body parts
    offset = {f"{i}_to_{j}": body_pose[i] - body_pose[j] for i, j in 
              [(14, 0), (15, 0), (16, 0), (17, 0), (3, 2), (4, 3), (6, 5), (7, 6), (9, 8), (10, 9), (12, 11), (13, 12)]}
    offset.update({"hand_left_to_4": hands[1] - body_pose[4], "hand_right_to_7": hands[0] - body_pose[7]})

    def apply_transformation(points, center, scale):
        M = cv2.getRotationMatrix2D(tuple(center), 0, scale)
        return warpAffine_kps(points, M)

    body_pose[0] = apply_transformation(body_pose[0], body_pose[1], scale_list[0])
    body_pose[[14, 15, 16, 17]] = apply_transformation(body_pose[[14, 15, 16, 17]], body_pose[0], scale_list[1])
    body_pose[[2, 5]] = apply_transformation(body_pose[[2, 5]], body_pose[1], scale_list[2])

    # Simplifying repetitive transformations by grouping transformations for arms and legs
    def transform_limb(start_idx, end_idx, scale_upper, scale_lower, hand_idx=None):
        body_pose[start_idx] = apply_transformation(body_pose[start_idx], body_pose[end_idx], scale_upper)
        body_pose[end_idx] = apply_transformation(body_pose[end_idx], body_pose[start_idx], scale_lower)
        if hand_idx is not None:
            hands[hand_idx] = apply_transformation(hands[hand_idx], body_pose[end_idx], scale_list[5])

    transform_limb(2, 3, scale_list[3], scale_list[4], hand_idx=1)
    transform_limb(5, 6, scale_list[3], scale_list[4], hand_idx=0)

    # Apply to legs similarly
    transform_limb(8, 9, scale_list[7], scale_list[8])
    transform_limb(11, 12, scale_list[7], scale_list[8])

    # Filter NaN and prepare the output
    pose_align = copy.deepcopy(pose_ori)
    for key in ['bodies', 'hands', 'faces']:
        pose_align[key] = np.nan_to_num(pose_ori[key], nan=-1.0)

    return pose_align

def process_video_frame(detector, img, refer_pose, align_args, offset, detect_resolution, image_resolution):
    _, pose_ori = detector(img, detect_resolution, image_resolution, output_type='cv2', return_pose_dict=True)
    pose_align = align_img(img, pose_ori, align_args, detect_resolution, image_resolution)
    pose_align['bodies']['candidate'] += offset
    pose_align['hands'] += offset
    pose_align['faces'] += offset

    # Scale back w to normalized coordinates
    pose_align['bodies']['candidate'][..., 0] /= img.shape[1] / img.shape[0]
    return pose_align

def run_align_video(args):
    video = cv2.VideoCapture(args.vidfn)
    total_frame, fps = video.get(cv2.CAP_PROP_FRAME_COUNT), video.get(cv2.CAP_PROP_FPS)

    detector = DWposeDetector(
        det_config=args.yolox_config, det_ckpt=args.yolox_ckpt, 
        pose_config=args.dwpose_config, pose_ckpt=args.dwpose_ckpt, keypoints_only=False).to('cuda' if torch.cuda.is_available() else 'cpu')

    refer_img = cv2.imread(args.imgfn_refer)
    _, pose_refer = detector(refer_img, args.detect_resolution, args.image_resolution, output_type='cv2', return_pose_dict=True)

    max_frame, align_frame = args.max_frame, args.align_frame
    video_frame_buffer, pose_list = [], []
    
    # Precompute alignment scales and offsets
    _, pose_1st_img = detector(refer_img, args.detect_resolution, args.image_resolution, output_type='cv2', return_pose_dict=True)
    align_args, offset = compute_alignment_parameters(pose_1st_img, pose_refer)

    for i in range(max_frame):
        ret, img = video.read()
        if not ret or i < align_frame:
            continue

        video_frame_buffer.append(img)
        pose_align = process_video_frame(detector, img, pose_refer, align_args, offset, args.detect_resolution, args.image_resolution)
        pose_list.append(pose_align)

    # Create output videos
    generate_output_videos(video_frame_buffer, pose_list, args.outfn, fps)

def compute_alignment_parameters(pose_1st_img, pose_refer):
    body_1st_img, body_ref_img = pose_1st_img['bodies']['candidate'], pose_refer['bodies']['candidate']

    # Calculate scales
    scale_neck = np.linalg.norm(body_ref_img[0] - body_ref_img[1]) / np.linalg.norm(body_1st_img[0] - body_1st_img[1])
    scale_face = np.linalg.norm(body_ref_img[16] - body_ref_img[17]) / np.linalg.norm(body_1st_img[16] - body_1st_img[17])
    offset = body_ref_img[1] - body_1st_img[1]

    align_args = {"scale_neck": scale_neck, "scale_face": scale_face}
    return align_args, offset

def generate_output_videos(video_frames, poses, outfn, fps):
    result_demo = [draw_pose(pose, *video_frame.shape[:2]) for pose, video_frame in zip(poses, video_frames)]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(result_demo, fps=fps)
    clip.write_videofile(outfn, fps=fps)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--detect_resolution', type=int, default=512)
    parser.add_argument('--image_resolution', type=int, default=720)
    parser.add_argument('--yolox_config', type=str, default="./pose/config/yolox_l_8xb8-300e_coco.py")
    parser.add_argument('--dwpose_config', type=str, default="./pose/config/dwpose-l_384x288.py")
    parser.add_argument('--yolox_ckpt', type=str, default="./pretrained_weights/dwpose/yolox_l_8x8_300e_coco.pth")
    parser.add_argument('--dwpose_ckpt', type=str, default="./pretrained_weights/dwpose/dw-ll_ucoco_384.pth")
    parser.add_argument('--align_frame', type=int, default=0)
    parser.add_argument('--max_frame', type=int, default=300)
    parser.add_argument('--imgfn_refer', type=str, default="./assets/images/0.jpg")
    parser.add_argument('--vidfn', type=str, default="./assets/videos/0.mp4")
    parser.add_argument('--outfn', type=str, default="./output_video.mp4")
    
    args = parser.parse_args()
    run_align_video(args)

if __name__ == '__main__':
    main()
