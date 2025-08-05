# Diffusion Classifier with Learnable Templates

This repository extends the [diffusion-classifier](https://github.com/diffusion-classifier/diffusion-classifier) project with learnable template functionality for improved text-conditional image classification using diffusion models.

## Overview

The project implements two main approaches:
1. **Standard Diffusion Classification**: Uses pre-defined text prompts to classify images based on diffusion model reconstruction errors
2. **Learnable Templates**: Automatically learns optimal text templates that improve classification performance

## Installation

Create a conda environment with the following command:
```bash
conda env create -f environment.yml
```

## Usage

### 1. Standard Diffusion Classification

Run classification using pre-defined prompts:

```bash
python diffusion_classifier.py \
    --dataset cifar10 \
    --split test \
    --prompt_path prompts/cifar10_prompts.csv \
    --to_keep 5 1 \
    --n_samples 50 500 \
    --loss l1 \
    --n_trials 1 \
    --samples_per_class 100
```

### 2. Training Learnable Templates

Train optimal templates for a specific dataset:

```bash
python diffusion_integration.py \
    --dataset cifar10 \
    --split train \
    --num_templates 8 \
    --learning_rate 1e-3 \
    --num_epochs 10 \
    --batch_size 4 \
    --samples_per_class 50 \
    --save_dir ./trained_templates
```


