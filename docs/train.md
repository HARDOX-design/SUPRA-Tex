# Training Overview

This repository provides inference code and pretrained models. This overview
shares the SUPRA-Tex training workflow and the open-source work it builds on, to
help others understand the method and conduct related research.

The workflow consists of data preparation, image-to-material prediction, and
observation-guided 3D material completion.

## 1. Data Preparation: Following Pixal3D

Our data processing workflow follows
[Pixal3D](https://github.com/TencentARC/Pixal3D). Starting from 3D assets with
PBR materials, we normalize the assets and render multiple views to obtain
aligned RGB images, material maps, and camera parameters.

We then use pretrained encoders to extract sparse 3D latents for geometry and
complete materials. Visible materials from a reference view are projected onto
the object surface to construct partial material observations. Geometry,
complete materials, and partial observations must share the same 3D coordinate
frame.

## 2. Image-to-Material Prediction: Built with DiffSynth-Studio

The 2D material prediction stage is implemented with
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), using
Qwen-Image-2.1 as the base model to jointly predict albedo, metallic, and
roughness from a reference image.

Training uses spatially aligned RGB images and material maps. We freeze the
pretrained backbone and adapt it to material prediction using LoRA and
learnable modality embeddings, with supervision on the material outputs in
latent space.

## 3. 3D Material Completion: Following TRELLIS.2

The 3D stage follows the sparse latent representation and Flow Matching
training approach of [TRELLIS.2](https://github.com/microsoft/TRELLIS.2),
with SUPRA-Tex's observation encoding and Interleaved Flow Transformer.

During training, we add noise to the complete material latents and learn to
predict the flow velocity, conditioned on geometry latents and partial material
observations. Observation tokens remain fixed to guide generation. The training
objective covers all valid surface locations, teaching the model to preserve
visible details while completing materials in unseen regions.

The 2D material predictor and 3D material completion model are trained
separately and connected during inference: materials are first predicted from
the input image, then encoded as 3D surface observations, and finally completed
into full PBR materials.
