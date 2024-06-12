import os,sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import av
import numpy as np
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
import glob
import torch.nn.functional as F

from musepose.models.pose_guider import PoseGuider
from musepose.models.unet_2d_condition import UNet2DConditionModel
from musepose.models.unet_3d import UNet3DConditionModel
from musepose.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from musepose.utils.util import get_fps, read_frames, save_videos_grid



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/test_stage_2.yaml")
    parser.add_argument("-W", type=int, default=768, help="Width")
    parser.add_argument("-H", type=int, default=768, help="Height")
    parser.add_argument("-L", type=int, default=300, help="video frame length")
    parser.add_argument("-S", type=int, default=48,  help="video slice frame number")
    parser.add_argument("-O", type=int, default=4,   help="video slice overlap frame number")

    parser.add_argument("--cfg",   type=float, default=3.5, help="Classifier free guidance")
    parser.add_argument("--seed",  type=int,   default=99)
    parser.add_argument("--steps", type=int,   default=20, help="DDIM sampling steps")
    parser.add_argument("--fps",   type=int)
    
    parser.add_argument("--skip",  type=int,   default=1, help="frame sample rate = (skip+1)") 
    args = parser.parse_args()

    print('Width:', args.W)
    print('Height:', args.H)
    print('Length:', args.L)
    print('Slice:', args.S)
    print('Overlap:', args.O)
    print('Classifier free guidance:', args.cfg)
    print('DDIM sampling steps :', args.steps)
    print("skip", args.skip)

    return args


def scale_video(video,width,height):
    video_reshaped = video.view(-1, *video.shape[2:])  # [batch*frames, channels, height, width]
    scaled_video = F.interpolate(video_reshaped, size=(height, width), mode='bilinear', align_corners=False)
    scaled_video = scaled_video.view(*video.shape[:2], scaled_video.shape[1], height, width)  # [batch, frames, channels, height, width]
    
    return scaled_video


def run_video_generation(
    config_path="./configs/test_stage_2.yaml", 
    width=768,
    height=768,
    length=300,
    slice_num=48,
    overlap=4, 
    cfg=3.5,
    seed=99,
    steps=20,
    fps=None,
    skip=1
):
    config = OmegaConf.load(config_path)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(seed)

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")

    def handle_single(ref_image_path,pose_video_path):
        print ('handle===',ref_image_path, pose_video_path)
        ref_name = Path(ref_image_path).stem
        pose_name = Path(pose_video_path).stem.replace("_kps", "")

        ref_image_pil = Image.open(ref_image_path).convert("RGB")

        pose_list = []
        pose_tensor_list = []
        pose_images = read_frames(pose_video_path)
        src_fps = get_fps(pose_video_path)
        print(f"pose video has {len(pose_images)} frames, with {src_fps} fps")
        L = min(length, len(pose_images))
        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )
        original_width,original_height = 0,0

        pose_images = pose_images[::skip+1]
        print("processing length:", len(pose_images))
        src_fps = src_fps // (skip + 1)
        print("fps", src_fps)
        L = L // ((skip + 1))
        
        for pose_image_pil in pose_images[: L]:
            pose_tensor_list.append(pose_transform(pose_image_pil))
            pose_list.append(pose_image_pil)
            original_width, original_height = pose_image_pil.size
            pose_image_pil = pose_image_pil.resize((width,height))

        # repeart the last segment
        last_segment_frame_num =  (L - slice_num) % (slice_num - overlap) 
        repeart_frame_num = (slice_num - overlap - last_segment_frame_num) % (slice_num - overlap) 
        for i in range(repeart_frame_num):
            pose_list.append(pose_list[-1])
            pose_tensor_list.append(pose_tensor_list[-1])

        
        ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
        ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=L)

        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0)

        video = pipe(
            ref_image_pil,
            pose_list,
            width,
            height,
            len(pose_list),
            steps,
            cfg,
            generator=generator,
            context_frames=slice_num,
            context_stride=1,
            context_overlap=overlap,
        ).videos


        m1 = config.pose_guider_path.split('.')[0].split('/')[-1]
        m2 = config.motion_module_path.split('.')[0].split('/')[-1]

        save_dir_name = f"{time_str}-{cfg}-{m1}-{m2}"
        save_dir = Path(f"./output/video-{date_str}/{save_dir_name}")
        save_dir.mkdir(exist_ok=True, parents=True)

        result = scale_video(video[:,:,:L], original_width, original_height)
        output_path1 = f"{save_dir}/{ref_name}_{pose_name}_{cfg}_{steps}_{skip}.mp4"
        save_videos_grid(
            result,
            output_path1,
            n_rows=1,
            fps=src_fps if fps is None else fps,
        )

        # print("ref_image_tensor size:", ref_image_tensor.size())
        # print("pose_tensor size:", pose_tensor[:, :, :L].size())
        # print("video size:", video[:, :, :L].size())

        # video = torch.cat([ref_image_tensor, pose_tensor[:,:,:L], video[:,:,:L]], dim=0) 
        # video = scale_video(video, original_width, original_height)     
        # output_path2 = f"{save_dir}/{ref_name}_{pose_name}_{cfg}_{steps}_{skip}_{m1}_{m2}.mp4"
        # save_videos_grid(
        #     video,
        #     output_path2,
        #     n_rows=3,
        #     fps=src_fps if fps is None else fps,
        # )
        
        # return { "output_path1": output_path1, "output_path2": output_path2 }
        return { "output_path": output_path1 }

    for ref_image_path_dir in config["test_cases"].keys():
        if os.path.isdir(ref_image_path_dir):
            ref_image_paths = glob.glob(os.path.join(ref_image_path_dir, '*.jpg'))
        else:
            ref_image_paths = [ref_image_path_dir]
        for ref_image_path in ref_image_paths:
            for pose_video_path_dir in config["test_cases"][ref_image_path_dir]:            
                if os.path.isdir(pose_video_path_dir):
                    pose_video_paths = glob.glob(os.path.join(pose_video_path_dir, '*.mp4'))
                else:
                    pose_video_paths = [pose_video_path_dir]
                for pose_video_path in pose_video_paths:
                    video_paths = handle_single(ref_image_path, pose_video_path)
                    all_video_paths.extend(video_paths)

    return all_video_paths


if __name__ == "__main__":
    args = parse_args()
    video_paths = run_video_generation(
        args.config, args.W, args.H, args.L, args.S, args.O, args.cfg, args.seed, args.steps, args.fps, args.skip
    )
    print(json.dumps(video_paths))
