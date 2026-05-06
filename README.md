# Attribute-Aware Consistency Prompt Learning

## Project Metadata
### Authors
- **Team:** Basel Alzahrani
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

## Introduction
Large-scale vision-language models such as CLIP have demonstrated strong zero-shot recognition ability by aligning image and text representations. However, manually designed prompts are often suboptimal for downstream tasks, especially under few-shot learning settings where only limited labeled samples are available.

Prompt learning methods such as CoOp and CoCoOp improve adaptation by learning context tokens while keeping the pretrained CLIP backbone frozen. Although effective, these methods often rely only on class names and may produce unstable predictions under low-data supervision.

This project proposes **Attribute-Aware Consistency Prompt Learning (AACPL)**, a lightweight extension of CoCoOp that improves prompt learning through richer semantic descriptions and dual-branch consistency regularization.

We:

- Enrich class prompts using semantic attributes.
- Learn two complementary prompt branches.
- Apply KL consistency loss between prompt branches.
- Use probability ensembling and temperature calibration at inference.

The goal is to improve few-shot classification accuracy while preserving the frozen visual encoder.


## Problem Statement
Few-shot prompt learning methods face several limitations:

### Q1: Limited Text Supervision
Using only class names may not fully exploit CLIP’s language prior.

### Q2: Prompt Instability
Different prompt variants may produce inconsistent predictions under few-shot training.

### Q3: Efficient Adaptation
Can we improve CoCoOp without modifying the visual backbone or adding heavy multimodal modules?

We evaluate whether semantic prompt enrichment and consistency learning can improve OxfordPets accuracy under the 16-shot setting.

## Application Area and Project Domain
This project belongs to:

- Computer Vision  
- Vision-Language Models  
- Prompt Learning  
- Few-Shot Learning  
- Transfer Learning  

Applications include:

- Fine-grained image recognition  
- Low-data domain adaptation  
- Efficient deployment of pretrained models  
- Lightweight adaptation for edge systems

## What is the paper trying to do, and what are you planning to do?
he reference paper CoCoOp proposes conditional prompt learning by generating instance-conditioned prompt tokens using image features.

This project extends that idea through:

Proposed AACPL Improvements
1. Attribute-Aware Prompt Construction

Instead of:
Persian We use: Persian, fluffy long-haired flat-faced cat

This gives richer semantic supervision.

2. Dual Prompt Branches

We use two prompt templates:

Template 1: a photo of a
Template 2: a blurry photo of a

Each branch predicts independently.

3. KL Consistency Regularization

Branch 2 is encouraged to match Branch 1 predictions using one-way KL divergence.

4. Improved Inference

At test time:

logits from both branches are averaged
temperature calibration is applied
final probabilities are returned

### Project Documents
- **Presentation PDF:** [Project Presentation](/presentation.PDF)
- **Presentation PPTX:** [Project Presentation](/presentation.pptx)
- **Term Paper PDF:** [Term Paper](/report.pdf)
- **Term Paper Latex Files:** [Term Paper Latex files](/report.zip)

### Reference Paper
- [Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557)
- 
### Reference GitHub
- [This is a reference Github](https://github.com/KaiyangZhou/CoOp)

### Reference Dataset
- [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)


## Project Technicalities

### Terminologies
### CLIP
Contrastive Language-Image Pretraining model for aligned image-text embeddings.

### Prompt Learning
Learning context tokens instead of full model fine-tuning.

### CoOp
Context Optimization with static learned prompts.

### CoCoOp
Conditional CoOp using image-conditioned prompts.

### Few-Shot Learning
Training with only a few labeled examples per class.

### Consistency Regularization
Encouraging multiple predictors to output similar predictions.

### Temperature Scaling
Calibrating logits before softmax.

### Problem Statements
### Problem 1
Class-name-only prompts underuse semantic language knowledge.

### Problem 2
Few-shot prompt learning may become unstable.

### Problem 3
Need stronger adaptation without expensive retraining.

### Loopholes or Research Areas
- Better prompt robustness under low-data settings.
- Better semantic exploitation of CLIP text encoder.
- Efficient multi-prompt learning.
- Generalization to unseen classes.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
### Idea 1: Attribute Prompts
Use semantic descriptors per class.

### Idea 2: Dual Prompt Agreement
Train multiple prompt branches to agree.

### Idea 3: Better Inference Calibration
Average branches and calibrate confidence.

### Proposed Solution: Code-Based Implementation
This repository modifies the original CoCoOp implementation.

## Main File Modified
trainers/cocoop.py

## Main Changes

- Added dual prompt branches
- Added KL consistency loss
- Added attribute-aware class prompts
- Added probability ensembling
- Added temperature calibration


### Key Components
### trainers/cocoop.py: Main training logic and prompt learner modifications.

### train.py: Training launcher.

### configs/trainers/CoCoOp/: Training configurations.

### configs/datasets/oxford_pets.yaml: Dataset configuration.

## Model Workflow
## Input

- OxfordPets image
- Class text prompts
- Few-shot labeled samples

## Training

1. Extract image features using frozen CLIP encoder.
2. Generate prompts from two branches.
3. Compute logits for both branches.
4. Apply CE loss to both.
5. Apply KL consistency loss.

## Output

- Final trained prompt learner
- Improved few-shot classifier

---

# Experimental Setup

- Dataset: OxfordPets
- Shots: 16-shot
- Classes: Base classes
- Backbone: CLIP ViT-B/16
- Seed: 1
- Epochs: 10

---

# Results

| Method | Accuracy |
|--------|----------|
| CoCoOp Baseline | 94.8% |
| Consistency Prompt Learning | 95.5% |
| AACPL (Final) | **96.0%** |

Improvement over baseline: **+1.2%**

---

# Ablation Study

| Lambda | Validation Accuracy |
|--------|---------------------|
| 10 | 98.7% |
| 20 | 98.7% |
| 30 | 98.7% |
| 40 | 98.7% |

The method remains stable across a wide range of consistency weights.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/KaiyangZhou/CoOp
    cd CoOp
    ```
2. **Replace File**
    replace existing trainers/cocoop.py with tho one in this repo
   
3. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```
4. **Dataset setup:**

Download the Oxford-IIIT Pet dataset and split file from CoOp and place it inside your dataset root directory.

Example:

```text
C:/Users/basel/Desktop/datasets/oxford_pets
```

5. **Training**
    Configure the training parameters in the provided configuration file and run:
    ```bash
        python train.py --root C:/Users/basel/Desktop/datasets --seed 1 --trainer CoCoOp --dataset-config-file configs/datasets/oxford_pets.yaml --config-file configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base

    ```

## Acknowledgments
Thanks to:

Dr. Muzammil Behzad for supervision and guidance.
Open-source contributors of CLIP, CoOp, and CoCoOp.
KFUPM for academic support.
