# Diffusion Classifier with Learnable Templates

This repository extends the [diffusion-classifier](https://github.com/diffusion-classifier/diffusion-classifier) project with learnable template functionality for improved text-conditional image classification using diffusion models.

## Overview
This extension enhances the original diffusion-classifier by replacing static text prompts with learnable template embeddings. Instead of using fixed prompts like "a photo of a [class]", the system learns optimal template representations that are specifically tuned for classification tasks using diffusion models.
The project implements two main approaches:
1. **Standard Diffusion Classification**: Uses pre-defined text prompts to classify images based on diffusion model reconstruction errors
2. **Learnable Templates**: Automatically learns optimal text templates that improve classification performance


## Project Structure

```
diffusion-classifier/
├── diffusion_classifier.py          # Original classification script (enhanced)
├── learnable_templates.py           # Core template learning modules
├── diffusion_integration.py         # Template training script  
├── diffusion/                       # Original diffusion utilities
│   ├── datasets.py
│   ├── models.py
│   └── utils.py
├── trained_templates/               # Saved template checkpoints
│   └── templates_[dataset]/
│       ├── best_templates.pt
│       ├── templates_epoch_*.pt
│       └── checkpoint_epoch_*.pt
└── logs/                           # Training and evaluation logs
```


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
#### Standard Diffusion Classification Parameters

| Parameter | Description |
|-----------|-------------|
| `--to_keep` | Number of classes to keep at each stage |
| `--n_samples` | Number of diffusion timesteps to sample at each stage |
| `--n_trials` | Number of times each sample is evaluated during the experiment |
| `--samples_per_class` | Create balanced test subset |

### 2. Training Learnable Templates

Train optimal templates for a specific dataset:

```bash
python diffusion_integration.py \
    --dataset cifar10 \
    --split train \
    --to_keep 5 1 \
    --n_samples 50 500 \
    --loss l1 \
    --n_trials 1 \
    --num_templates 8 \
    --learning_rate 1e-3 \
    --num_epochs 10 \
    --batch_size 4 \
    --samples_per_class 100 \
    --save_dir ./trained_templates
```

Evaluation with Learned Templates

```bash
python diffusion_classifier.py \
    --dataset cifar10 \
    --split test \
    --prompt_path prompts/cifar10_prompts.csv \
    --to_keep 5 1 \
    --n_samples 50 500 \
    --loss l1 \
    --n_trials 1 \
    --samples_per_class 100 \
    --template_path ./trained_templates/templates_pets/best_templates.pt
```
