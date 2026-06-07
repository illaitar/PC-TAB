# PC-TAB and PC-TABD

**PC-TAB** is a physically controllable framework for motion blur synthesis. It turns ordinary sharp video into realistic blurred data and stores the per-pixel motion trajectories that produced each blur.

**PC-TABD** is the dataset generated with PC-TAB. It contains 21,500 paired blurred videos with sharp references, synthesis-defined trajectories, generation metadata, and train and test splits.

## Why

Motion blur is a common failure mode for visual models, but paired sharp and blurred data is expensive to capture and often tied to a narrow camera setup. PC-TAB treats blur synthesis as a controllable data engine. It preserves labels while changing only image appearance, so the same source data can train models that are more robust to real motion blur.

## Method

PC-TAB starts from sharp video triplets, separates global camera motion from object residual motion, samples interpretable physical factors, and integrates exposure in linear light. The pipeline also models visibility, occlusions, rolling shutter, sensor noise, nonlinear response, and sharpening.

The framework can be used for on-the-fly augmentation or for generating a fixed benchmark dataset. In the paper, PC-TAB improves six deblurring architectures on GoPro and REDS, transfers to RealBlur-J and RealBlur-R, and improves object detection under real motion blur on BDD100K.

## Dataset

Dataset: https://huggingface.co/datasets/Illaitar/PC-TABD

Code: https://github.com/illaitar/PC-TABD

## Quickstart

Install the repository dependencies:

```bash
git clone https://github.com/illaitar/PC-TABD
cd PC-TABD
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download PC-TABD from Hugging Face and place it under `data/pc_tabd`.

## Repository Layout

`pc-tab` contains the core synthesis code for trajectories, visibility, shutter integration, ISP modeling, and camera motion estimation.

`scripts` contains helper scripts for GoPro processing, optical flow and depth precomputation, runtime checks, and trajectory analysis.

## License

This project is licensed under **Creative Commons Attribution 4.0 International**.
