Project Title:
PromptRobust-CLIP: Evaluating Vision-Language Model Sensitivity to Text Prompt Variations

Project Idea:
Vision-Language Models (VLMs) such as CLIP rely heavily on textual prompts for zero-shot image classification. However, small changes in prompt wording may significantly affect model predictions. This project investigates the robustness and stability of CLIP outputs under systematic prompt variations, including paraphrasing, descriptive modifiers, stylistic changes, and length differences. We will evaluate how prompt changes influence classification accuracy, confidence scores, and label consistency. By quantifying label flip rates and prediction variance, we aim to measure the sensitivity of CLIP to prompt perturbations. The results will provide insights into the reliability of prompt-based inference in modern VLMs.

Datasets to be Used:
1.	ImageNet-1K (ILSVRC 2012 Validation Set)
o	50,000 labeled images
o	1,000 object classes
o	Used for zero-shot evaluation of CLIP with different prompt templates
Backup:
2.	MS COCO (Common Objects in Context)
o	Diverse real-world images
o	Used for additional robustness analysis
