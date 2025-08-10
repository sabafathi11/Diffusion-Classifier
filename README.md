# Diffusion Classifier with Learnable Templates

This repository extends the [diffusion-classifier](https://github.com/diffusion-classifier/diffusion-classifier) project with learnable template functionality for improved text-conditional image classification using diffusion models.
This repository also contains two complementary approaches for improving diffusion-based image classification through hierarchical clustering: **Hierarchical Clustering** and **Beam Search**. Both methods leverage CLIP embeddings to organize class labels into semantic hierarchies and reduce computational costs during inference.

## Overview
This extension enhances the original diffusion-classifier by replacing static text prompts with learnable template embeddings. Instead of using fixed prompts like "a photo of a [class]", the system learns optimal template representations that are specifically tuned for classification tasks using diffusion models.
The project implements two main approaches:
1. **Standard Diffusion Classification**: Uses pre-defined text prompts to classify images based on diffusion model reconstruction errors
2. **Learnable Templates**: Automatically learns optimal text templates that improve classification performance
Moreover traditional diffusion-based classification evaluates all classes simultaneously, which can be computationally expensive for datasets with many classes. This project introduces two hierarchical approaches:

1. **Hierarchical Clustering** : Uses a tree-based approach where classification decisions are made level by level, progressively narrowing down candidates.
2. **Beam Search** : Maintains multiple candidate hypotheses (beams) at each hierarchical level and selects the top-k most promising paths.

Both approaches use CLIP embeddings to build semantic hierarchies of class labels, allowing for more efficient and interpretable classification.

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
└── data/                           # Training and evaluation logs
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
    --num_templates 4 \
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

### 3. Hierarchical Clustering Approach
```bash
python clustering_diffusion_classifier.py \
    --dataset cifar10 \
    --split test \
    --n_trials 1 \
    --loss l1 \
    --samples_per_class 10 \
    --prompt_path prompts/cifar10_prompts.csv \
    --use_clustering \
    --cluster_depth 3
```

### 4. Beam Search Approach
```bash
python python beam_search_diffusion_classifier.py \
    --dataset cifar10 \
    --split test \
    --n_trials 1 \
    --loss l1 \
    --samples_per_class 10 \
    --prompt_path prompts/cifar10_prompts.csv \
    --use_clustering \
    --beam_width 2
```

## How It Works

### 1. Hierarchical Tree Construction
Both approaches start by building a semantic hierarchy:

1. **Extract Class Names**: Parse unique class names from the prompt CSV
2. **CLIP Embedding**: Generate CLIP text embeddings for each class name
3. **Hierarchical Clustering**: Use agglomerative clustering to build a tree structure
4. **Centroid Calculation**: For each internal node, identify the centroid class (most representative)

### 2. Hierarchical Evaluation

#### Hierarchical Clustering (Greedy)
- Start from the root with all classes
- At each depth level:
  - Get clusters containing current candidates
  - Evaluate representative classes from each cluster
  - Select the best cluster (lowest diffusion error)
  - Continue with classes from the selected cluster
- Final evaluation on remaining candidates

#### Beam Search (Multiple Hypotheses)
- Maintain multiple candidate paths (beams) simultaneously
- At each depth level:
  - For each beam, evaluate cluster representatives
  - Generate new beams from all possible cluster choices
  - Keep top-k beams based on cumulative scores
- Final evaluation combines candidates from all surviving beams


## Output Files

Each run generates comprehensive analysis in the output folder:

- `confusion_matrix.png`: Visual confusion matrix
- `classification_report.txt`: Detailed per-class metrics
- `results_summary.txt`: Overall performance summary
- `hierarchical_tree_labels_clip.pkl`: Cached tree structure
- `depth_error_histogram.png`: Distribution of errors by hierarchical depth (hierarchical clustering only)
- `depth_error_stats.txt`: Detailed depth error analysis (hierarchical clustering only)

