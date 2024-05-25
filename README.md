# MusePose

MusePose: a Pose-Driven Image-to-Video Framework for Virtual Human Generation
</br>
Zhengyan Tong,
Chao Li,
Zhaokang Chen,
Bin Wu<sup>†</sup>,
Wenjiang Zhou
(<sup>†</sup>Corresponding Author, benbinwu@tencent.com)

**[github](https://github.com/TMElyralab/MusePose)**    **[huggingface](https://huggingface.co/TMElyralab/MusePose)**    **space**    **Project (comming soon)**

`MusePose` is an image-to-video generation framework for virtual human under control signal like pose. We believe that generating virtual human performing different actions is crutial in many scenarios. Together with [MuseV](https://github.com/TMElyralab/MuseV) and [MuseTalk](https://github.com/TMElyralab/MuseTalk), we expect that MusePose can contribute to our virtual human solution to, towards the path to more realistic generation.

Here, we introduce the first version of MusePose, `MusePose-v1`. We really thank [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone) for their technical report and [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) for their code base, which significant expedite the development of the community and MusePose.

# Overview
`MusePose-v1` is a diffusion-based virtual human video generation framework, which

1. generates video of the character in a reference image under the pose given by a reference pose video
1. `pose aligment` code that **SIGNIFICANTLY** helps to maintain the character in an image. It modifies the pose of an input video given the character in an image.
1. checkpoint available trained on the UBC fashion video dataset and an internal authorized dataset, with an resotion of `512 x 512` and `48` frames.
1. training codes.


# News
- [05/24/2024] Release MusePose project and pretrained models.

## Cases
<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td width="60%">Results</td>
        <td width="40%">Methods</td>
  </tr>
  <tr>
    <td >
      <video src=assets/video/video1.mp4 controls preload></video>
    </td>
    <td >
      Pose Align + MusePose
    </td>
  </tr>
  <tr>
    <td >
      <video src=assets/video/video2.mp4 controls preload></video>
    </td>
    <td >
      Pose Align + MusePose + FaceFusion
    </td>
  </tr>
</table >


# TODO:
- [x] trained models and inference codes of MusePose-v1.
- [x] pose alignment codes.
- [x] training codes.
- [ ] Huggingface Gradio demo.
- [ ] a improved architecture and model (may take longer).


# Getting Started
We provide a detailed tutorial about the installation and the basic usage of MusePose for new users:

## Installation
To prepare the Python environment and install additional packages such as opencv, diffusers, mmcv, etc., please follow the steps below:

### Build environment

We recommend a python version >=3.10 and cuda version =11.7. Then build environment as follows:

```shell
pip install -r requirements.txt
```

### mmlab packages
```bash
pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0" 
```

### Download ffmpeg-static
Download the ffmpeg-static and
```
export FFMPEG_PATH=/path/to/ffmpeg
```
for example:
```
export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static
```
### Download weights
You can download weights manually as follows:

1. Download our trained [weights](https://huggingface.co/TMElyralab/MusePose).

2. Download the weights of other components:
   - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
   - [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
   - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)

Finally, these weights should be organized in `models` as follows:
```
./pretrained_weights/
|-- MusePose
|   └── denoising_unet.pth
|   └── motion_module.pth
|   └── pose_guider.pth
|   └── reference_unet.pth
|-- dwpose
|   └── dw-ll_ucoco_384.pth
|-- image_encoder
|   |-- config.json
|   |-- pytorch_model.bin
|-- sd-vae-ft-mse
    |-- config.json
    |-- diffusion_pytorch_model.bin

```
## Quickstart

### Pose Alignment


### Inference
Here, we provide the inference script. 
```
python -m scripts.inference --inference_config configs/inference/test.yaml 
```
configs/inference/test.yaml is the path to the inference configuration file, including video_path and audio_path.
The video_path should be either a video file, an image file or a directory of images.

### Face Enhancement

If you want to enhance the face region to have a better consistency of the face, you could use [FaceFusion](https://github.com/facefusion/facefusion). You can use the `face-swap` function to swap the face in the reference image to the generated video.

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
