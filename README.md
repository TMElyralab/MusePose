# MusePose

MusePose: a Pose-Driven Image-to-Video Framework for Virtual Human Generation. 

Zhengyan Tong,
Chao Li,
Zhaokang Chen,
Bin Wu<sup>†</sup>,
Wenjiang Zhou
(<sup>†</sup>Corresponding Author, benbinwu@tencent.com)

Lyra Lab, Tencent Music Entertainment


**[github](https://github.com/TMElyralab/MusePose)**    **[huggingface](https://huggingface.co/TMElyralab/MusePose)**    **space (comming soon)**    **Project (comming soon)**    **Technical report (comming soon)**

[MusePose](https://github.com/TMElyralab/MusePose) is an image-to-video generation framework for virtual human under control signal such as pose. The current released model was an implementation of [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone) by optimizing [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone).

`MusePose` is the last building block of **the Muse opensource serie**. Together with [MuseV](https://github.com/TMElyralab/MuseV) and [MuseTalk](https://github.com/TMElyralab/MuseTalk), we hope the community can join us and march towards the vision where a virtual human can be generated end2end with native ability of full body movement and interaction. Please stay tuned for our next milestone!

We really appreciate [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone) for their academic paper and [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) for their code base, which have significantly expedited the development of the AIGC community and [MusePose](https://github.com/TMElyralab/MusePose).

Update:
1. We support [Comfyui-MusePose](https://github.com/TMElyralab/Comfyui-MusePose) now!

## Overview
[MusePose](https://github.com/TMElyralab/MusePose) is a diffusion-based and pose-guided virtual human video generation framework.  
Our main contributions could be summarized as follows:
1. The released model can generate dance videos of the human character in a reference image under the given pose sequence. The result quality exceeds almost all current open source models within the same topic.
2. We release the `pose align` algorithm so that users could align arbitrary dance videos to arbitrary reference images, which **SIGNIFICANTLY** improved inference performance and enhanced model usability.
3. We have fixed several important bugs and made some improvement based on the code of [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone).

## Demos
<table class="center">
    
<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/TMElyralab/MusePose/assets/47803475/bb52ca3e-8a5c-405a-8575-7ab42abca248" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/TMElyralab/MusePose/assets/47803475/6667c9ae-8417-49a1-bbbb-fe1695404c23" muted="false"></video>
    </td>
</tr>

<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/TMElyralab/MusePose/assets/47803475/7f7a3aaf-2720-4b50-8bca-3257acce4733" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/TMElyralab/MusePose/assets/47803475/c56f7e9c-d94d-494e-88e6-62a4a3c1e016" muted="false"></video>
    </td>
</tr>


<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/TMElyralab/MusePose/assets/47803475/00a9faec-2453-4834-ad1f-44eb0ec8247d" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/TMElyralab/MusePose/assets/47803475/41ad26b3-d477-4975-bf29-73a3c9ed0380" muted="false"></video>
    </td>
</tr>

<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/TMElyralab/MusePose/assets/47803475/2bbebf98-6805-4f1b-b769-537f69cc0e4b" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/TMElyralab/MusePose/assets/47803475/1b2b97d0-0ae9-49a6-83ba-b3024ae64f08" muted="false"></video>
    </td>
</tr>

</table>


## News
- [05/27/2024] Release `MusePose` and pretrained models.
- [05/31/2024] Support [Comfyui-MusePose](https://github.com/TMElyralab/Comfyui-MusePose)

## Todo:
- [x] release our trained models and inference codes of MusePose.
- [x] release pose align algorithm.
- [x] Comfyui-MusePose
- [ ] training guidelines.
- [ ] Huggingface Gradio demo.
- [ ] a improved architecture and model (may take longer).


# Getting Started
We provide a detailed tutorial about the installation and the basic usage of MusePose for new users:

## Installation
To prepare the Python environment and install additional packages such as opencv, diffusers, mmcv, etc., please follow the steps below:

### Build environment

We recommend a python version >=3.10 and cuda version==12.1. Then build environment as follows:

```shell
pip install -r requirements.txt
```


### Download weights
You can download weights manually as follows:

1. Download our trained [weights](https://huggingface.co/TMElyralab/MusePose).

2. Download the weights of other components:
   - [sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/unet)
   - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
   - [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
   - [yolox](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth) - Make sure to rename to `yolox_l_8x8_300e_coco.pth`
   - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)

Finally, these weights should be organized in `pretrained_weights` as follows:
```
./pretrained_weights/
|-- MusePose
|   |-- denoising_unet.pth
|   |-- motion_module.pth
|   |-- pose_guider.pth
|   └── reference_unet.pth
|-- dwpose
|   |-- dw-ll_ucoco_384.pth
|   └── yolox_l_8x8_300e_coco.pth
|-- sd-image-variations-diffusers
|   └── unet
|       |-- config.json
|       └── diffusion_pytorch_model.bin
|-- image_encoder
|   |-- config.json
|   └── pytorch_model.bin
└── sd-vae-ft-mse
    |-- config.json
    └── diffusion_pytorch_model.bin

```
## Quickstart
### Inference
#### Preparation
Prepare your referemce images and dance videos in the folder ```./assets``` and organnized as the example: 
```
./assets/
|-- images
|   └── ref.png
└── videos
    └── dance.mp4
```

#### Pose Alignment
Get the aligned dwpose of the reference image:
```
python pose_align.py --imgfn_refer ./assets/images/ref.png --vidfn ./assets/videos/dance.mp4
```
After this, you can see the pose align results in ```./assets/poses```, where ```./assets/poses/align/img_ref_video_dance.mp4``` is the aligned dwpose and the ```./assets/poses/align_demo/img_ref_video_dance.mp4``` is for debug.

#### Inferring MusePose
Add the path of the reference image and the aligned dwpose to the test config file ```./configs/test_stage_2.yaml``` as the example:
```
test_cases:
  "./assets/images/ref.png":
    - "./assets/poses/align/img_ref_video_dance.mp4"
```

Then, simply run
```
python test_stage_2.py --config ./configs/test_stage_2.yaml
```
```./configs/test_stage_2.yaml``` is the path to the inference configuration file.

Finally, you can see the output results in ```./output/```

##### Reducing VRAM cost
If you want to reduce the VRAM cost, you could set the width and height for inference. For example,
```
python test_stage_2.py --config ./configs/test_stage_2.yaml -W 512 -H 512
```
It will generate the video at 512 x 512 first, and then resize it back to the original size of the pose video.

Currently, it takes 16GB VRAM to run on 512 x 512 x 48 and takes 28GB VRAM to run on 768 x 768 x 48. However, it should be noticed that the inference resolution would affect the final results (especially face region).

#### Face Enhancement

If you want to enhance the face region to have a better consistency of the face, you could use [FaceFusion](https://github.com/facefusion/facefusion). You could use the `face-swap` function to swap the face in the reference image to the generated video.

### Training



# Acknowledgement
1. We thank [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone) for their technical report, and have refer much to [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) and [diffusers](https://github.com/huggingface/diffusers).
1. We thank open-source components like [AnimateDiff](https://animatediff.github.io/), [dwpose](https://github.com/IDEA-Research/DWPose), [Stable Diffusion](https://github.com/CompVis/stable-diffusion), etc.. 

Thanks for open-sourcing!

# Limitations
- Detail consitency: some details of the original character are not well preserved (e.g. face region and complex clothing).
- Noise and flickering: we observe noise and flicking in complex background. 

# Citation
```bib
@article{musepose,
  title={MusePose: a Pose-Driven Image-to-Video Framework for Virtual Human Generation},
  author={Tong, Zhengyan and Li, Chao and Chen, Zhaokang and Wu, Bin and Zhou, Wenjiang},
  journal={arxiv},
  year={2024}
}
```
# Disclaimer/License
1. `code`: The code of MusePose is released under the MIT License. There is no limitation for both academic and commercial usage.
1. `model`: The trained model are available for non-commercial research purposes only.
1. `other opensource model`: Other open-source models used must comply with their license, such as `ft-mse-vae`, `dwpose`, etc..
1. The testdata are collected from internet, which are available for non-commercial research purposes only.
1. `AIGC`: This project strives to impact the domain of AI-driven video generation positively. Users are granted the freedom to create videos using this tool, but they are expected to comply with local laws and utilize it responsibly. The developers do not assume any responsibility for potential misuse by users.
