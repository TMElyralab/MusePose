# MusePose

### MusePose: a Pose-Driven Image-to-Video Framework for Virtual Human Generation ###
</br>
Zhengyan Tong,
Chao Li,
Zhaokang Chen,
Bin Wu<sup>†</sup>,
Wenjiang Zhou
(<sup>†</sup>Corresponding Author, benbinwu@tencent.com)

[MusePose](https://github.com/TMElyralab/MusePose) is an image-to-video generation framework for virtual human under control signal like pose. We believe that generating virtual human performing different actions is crutial in many scenarios. Together with [MuseV](https://github.com/TMElyralab/MuseV) and [MuseTalk](https://github.com/TMElyralab/MuseTalk), we expect that [MusePose](https://github.com/TMElyralab/MusePose) can contribute to our virtual human solution, moving towards more wonderful AIGC in the future.

Here, we present the first version of **MusePose**, `MusePose-v1`. We really thank [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone) for their academic paper and [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) for their code base, which significant expedite the development of the AIGC community and [MusePose](https://github.com/TMElyralab/MusePose).

## Overview
[MusePose](https://github.com/TMElyralab/MusePose) is a diffusion-based and pose-guided virtual human video generation framework.  
Our main contributions could be summarized as follows:
1. The released model can generate dance videos of the human character in a reference image under the given pose sequence, and the result quality exceeds almost all current open source models within the same topic.
2. We released the `pose align` algorithm so that users could align arbitray dance videos as their pose sequence to arbitray reference image, which both **SIGNIFICANTLY** improved inference performance and enhanced model usability.
3. We fixed several serious bugs and made some improvement based on the code of (https://github.com/MooreThreads/Moore-AnimateAnyone).
4. [huggingface](https://huggingface.co/TMElyralab/MusePose) is comming soon.

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
        <video controls autoplay loop src="https://github.com/TMElyralab/MusePose/assets/47803475/64d14512-a1db-469b-8021-2ae817d2f729" muted="false"></video>
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
        <video controls autoplay loop src="https://github.com/TMElyralab/MusePose/assets/47803475/866d54b0-bad3-413b-9982-3518fd6c5de8" muted="false"></video>
    </td>
</tr>

</table>


## News
- [05/27/2024] Release `MusePose-v1` and pretrained models.


## Todo:
- [x] release our trained models and codes of MusePose-v1.
- [x] release pose align algorithm.
- [ ] training guidelines.
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


### Download weights
You can download weights manually as follows:

1. Download our trained [weights](https://huggingface.co/TMElyralab/MusePose).

2. Download the weights of other components:
   - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
   - [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
   - [yolox](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth)
   - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)

Finally, these weights should be organized in `models` as follows:
```
./pretrained_weights/
|-- MusePose
|   |-- denoising_unet.pth
|   |-- motion_module.pth
|   |-- pose_guider.pth
|   |-- reference_unet.pth
|-- dwpose
|   |-- dw-ll_ucoco_384.pth
    |-- yolox_l_8x8_300e_coco.pth
|-- image_encoder
|   |-- config.json
|   |-- pytorch_model.bin
|-- sd-vae-ft-mse
    |-- config.json
    |-- diffusion_pytorch_model.bin

```
## Quickstart

### Prepare 
prepare your referemce images and dance videos in the folder ```./asserts``` and organnized as the example: 
```
./asserts/
|-- images
|   |-- ref.jpg
|-- videos
|   |-- dance.mp4
```

### Pose Alignment
Get the aligned dwpose of the reference image:
```
python pose_align.py --imgfn_refer ./assets/images/ref.jpg --vidfn ./assets/videos/dance.mp4
```
After this, you can see the pose align results in ```./assets/poses```, where ```./assets/poses/align/img_ref_video_dance.mp4``` is the aligned dwpose and the ```./assets/poses/align_demo/img_ref_video_dance.mp4``` is for debug.

Add the path of the reference image and the aligned dwpose to the test config file ```./configs/test_stage_2.yaml``` as the example:
```
test_cases:
  "./assets/images/ref.jpg":
    - "./assets/poses/align/img_ref_video_dance.mp4"
```

### Inference
Here, we provide the inference script. 
```
python test_stage_2.py --config ./configs/test_stage_2.yaml
```
```./configs/test_stage_2.yaml``` is the path to the inference configuration file.

Finally, you can see the output results in ```./output/```
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
