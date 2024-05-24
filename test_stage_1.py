import os,sys
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List
import glob

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

from configs.prompts.test_cases import TestCasesDict
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2img import Pose2ImagePipeline
from src.utils.util import get_fps, read_frames, save_videos_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",default="./configs/test_stage_1.yaml")
    parser.add_argument("-W", type=int, default=768)
    parser.add_argument("-H", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cnt", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=7)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--fps", type=int)
    args = parser.parse_args()

    return args



def main():
    args = parse_args()

    config = OmegaConf.load(args.config)

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
        # config.motion_module_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)


    width, height = args.W, args.H

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

    pipe = Pose2ImagePipeline(
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

    m1 = config.pose_guider_path.split('.')[0].split('/')[-1]
    save_dir_name = f"{time_str}-{m1}"

    save_dir = Path(f"./output/image-{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    def handle_single(ref_image_path, pose_path,seed):
        generator = torch.manual_seed(seed)
        ref_name = Path(ref_image_path).stem
        # pose_name = Path(pose_image_path).stem.replace("_kps", "")
        pose_name = Path(pose_path).stem

        ref_image_pil = Image.open(ref_image_path).convert("RGB")
        pose_image = Image.open(pose_path).convert("RGB")

        original_width, original_height = pose_image.size

        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )

        pose_image_tensor = pose_transform(pose_image)
        pose_image_tensor = pose_image_tensor.unsqueeze(0)  # (1, c, h, w)

        ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)

        image = pipe(
            ref_image_pil,
            pose_image,
            width,
            height,
            args.steps,
            args.cfg,
            generator=generator,
        ).images

        image = image.squeeze(2).squeeze(0)  # (c, h, w)
        image = image.transpose(0, 1).transpose(1, 2)  # (h w c)
        #image = (image + 1.0) / 2.0  # -1,1 -> 0,1
        
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image, 'RGB')
        # image.save(os.path.join(save_dir, f"{ref_name}_{pose_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.png"))

        image_grid = Image.new('RGB',(original_width*3,original_height))
        imgs = [ref_image_pil,pose_image,image]
        x_offset = 0
        for img in imgs:
            img = img.resize((original_width*2, original_height*2))
            img.save(os.path.join(save_dir, f"res_{ref_name}_{pose_name}_{args.cfg}_{seed}.jpg"))
            img = img.resize((original_width,original_height))
            image_grid.paste(img, (x_offset,0))
            x_offset += img.size[0]
        image_grid.save(os.path.join(save_dir, f"grid_{ref_name}_{pose_name}_{args.cfg}_{seed}.jpg"))


    for ref_image_path_dir in config["test_cases"].keys():
        if os.path.isdir(ref_image_path_dir):
            ref_image_paths = glob.glob(os.path.join(ref_image_path_dir, '*.jpg'))
        else:
            ref_image_paths = [ref_image_path_dir]
        for ref_image_path in ref_image_paths:
            for pose_image_path_dir in config["test_cases"][ref_image_path_dir]:            
                if os.path.isdir(pose_image_path_dir):
                    pose_image_paths = glob.glob(os.path.join(pose_image_path_dir, '*.jpg'))
                else:
                    pose_image_paths = [pose_image_path_dir]
                for pose_image_path in pose_image_paths:
                    for i in range(args.cnt):
                        handle_single(ref_image_path, pose_image_path, args.seed + i) 


if __name__ == "__main__":
    main()
    