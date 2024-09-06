import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from pose.script.tool import save_videos_from_pil
from pose.script.dwpose import draw_pose


def get_video_info(video_path):
    """Retrieve video properties such as width, height, and fps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(np.around(cap.get(cv2.CAP_PROP_FPS)))
    cap.release()
    return width, height, fps


def draw_dwpose(video_path, pose_path, out_path, draw_face):
    # Get video properties
    width, height, fps = get_video_info(video_path)

    # Calculate render and save dimensions
    k_render = 1024 / min(width, height)
    h_render, w_render = int(k_render * height // 2 * 2), int(k_render * width // 2 * 2)
    k_save = 768 / min(width, height)
    h_save, w_save = int(k_save * height // 2 * 2), int(k_save * width // 2 * 2)

    # Load pose data
    poses = np.load(pose_path, allow_pickle=True).tolist()

    frames = []
    for pose in poses:
        detected_map = draw_pose(pose, h_render, w_render, draw_face)
        detected_map = cv2.resize(detected_map, (w_save, h_save), interpolation=cv2.INTER_AREA)
        detected_map = cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(detected_map))

    # Save the generated frames as a video
    save_videos_from_pil(frames, out_path, fps)


def process_video(video_path, video_dir, pose_dir, save_dir, draw_face):
    """Processes a single video by drawing poses and saving the output."""
    video_name = os.path.relpath(video_path, video_dir)
    base_name = os.path.splitext(video_name)[0]
    
    pose_path = os.path.join(pose_dir, base_name + '.npy')
    if not os.path.exists(pose_path):
        print(f'No keypoint file found for: {pose_path}')
        return

    out_path = os.path.join(save_dir, base_name + '.mp4')
    if os.path.exists(out_path):
        print(f'Already rendered pose video: {out_path}')
        return

    draw_dwpose(video_path, pose_path, out_path, draw_face)
    print(f"Processed: {video_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="./UBC_fashion/test", help='Path to the directory containing video files')
    parser.add_argument("--pose_dir", type=str, help='Directory containing pose keypoints; auto-created if not provided')
    parser.add_argument("--save_dir", type=str, help='Directory to save output videos; auto-created if not provided')
    parser.add_argument("--draw_face", type=bool, default=False, help='Whether to draw face keypoints or not')
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for parallel video processing")
    args = parser.parse_args()

    video_dir = args.video_dir
    pose_dir = args.pose_dir or f"{video_dir}_dwpose_keypoints"
    save_dir = args.save_dir or f"{video_dir}_dwpose" if args.draw_face else f"{video_dir}_dwpose_without_face"

    # Create output directory if not exists
    os.makedirs(save_dir, exist_ok=True)

    # Collect all video files from the directory
    video_mp4_paths = [os.path.join(root, file) for root, _, files in os.walk(video_dir) for file in files if file.endswith(".mp4")]
    print(f"Found {len(video_mp4_paths)} video(s)")

    # Process videos using ThreadPoolExecutor for parallelism
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_video, video_path, video_dir, pose_dir, save_dir, args.draw_face) for video_path in video_mp4_paths]
        
        # Using tqdm to track the progress of video processing
        for future in tqdm(futures, total=len(video_mp4_paths), desc="Processing videos"):
            future.result()

    print('All videos processed successfully!')


if __name__ == "__main__":
    main()
