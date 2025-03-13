# PIC2D2L: Precision and Interpretability in Colorectal Cancer Diagnosis with Deep Learning

## Overview

This repository contains the implementation of a novel framework that utilizes YOLOv8-based deep learning models to
recognize and interpret colorectal cancer cases.

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0
- Ultralytics YOLOv8

### Setup

1. Clone the repository:

```bash
   git clone https://github.com/HossamBalaha/PIC2D2L-Precision-and-Interpretability-in-Colorectal-Cancer-Diagnosis-with-Deep-Learning
   cd PIC2D2L-Precision-and-Interpretability-in-Colorectal-Cancer-Diagnosis-with-Deep-Learning
```

2. Install dependencies:

```bash
   pip install -r requirements.txt
```

## Description

The provided code encompasses several key components that contribute to the development and evaluation of a deep
learning-based framework for colorectal cancer diagnosis.
Below is a detailed description of the code in structured paragraphs:

The code begins by setting up the environment and preparing the dataset for training and evaluation. It utilizes the
splitfolders library to divide the dataset (e.g., CK+48) into training, validation, and testing subsets with a
predefined ratio (70% training, 15% validation, and 15% testing). This ensures a balanced distribution of data across
different phases of the machine learning pipeline. The dataset directory structure is organized such that each class (
emotion) has its own folder, and the splitting process maintains this structure. The input images are resized to a
uniform shape of 100x100 pixels to standardize the data for model training.

The core of the implementation revolves around training various YOLOv8 classification models (yolov8n, yolov8s, yolov8m,
yolov8l, yolov8x). These models are initialized using pre-trained weights (-cls.pt) and fine-tuned on the prepared
dataset. The training process involves specifying parameters such as the number of epochs (set to 250), image size, and
enabling options for saving plots and results. Each model variant is trained independently, and the performance
metrics (top-1 and top-5 accuracy) are logged after validation. The modular design allows for easy experimentation with
different model architectures and hyperparameters.

A critical component of the code is the CalculateMetrics function, which computes a comprehensive set of evaluation
metrics based on the confusion matrix derived from the model's predictions. These metrics include Accuracy, Precision,
Recall, Specificity, F1-score, Intersection over Union (IoU), Balanced Accuracy (BAC), and Matthews Correlation
Coefficient (MCC). The function handles multi-class classification scenarios and calculates weighted averages for each
metric, accounting for class imbalance in the dataset. This ensures that the evaluation reflects the model's performance
across all classes, providing a holistic view of its effectiveness.

The code aggregates results from multiple experiments by iterating through CSV files containing predictions and ground
truth labels for the test dataset. For each model variant, it reads the corresponding CSV file, extracts the actual and
predicted labels, and computes the evaluation metrics using the CalculateMetrics function. The results are compiled into
a Pandas DataFrame, which is then formatted into a LaTeX table for easy inclusion in academic publications. This
systematic approach to result aggregation facilitates comparative analysis of different model architectures and their
respective performances.

The code incorporates several utilities to ensure smooth execution. For instance, it addresses potential multiprocessing
issues by setting the KMP_DUPLICATE_LIB_OK environment variable to "TRUE," preventing conflicts related to OpenMP
libraries. Additionally, warnings are suppressed to maintain clean output during execution. These considerations reflect
attention to practical challenges that may arise during model development and evaluation.

## Materials

The framework obtained ten anonymized H&E-stained colorectal cancer (CRC) tissue slides from the pathology repository at
University Medical Center Mannheim, part of Heidelberg University in Mannheim, Germany, to study eight distinct tissue
categories: tumor epithelium, simple stroma, complex stroma, immune cells, debris, normal mucosal glands, adipose
tissue, and background. Through manual annotation and tessellation of tissue areas, 5,000 representative images were
generated to form the training and testing dataset for the classification problem, ensuring a comprehensive depiction of
CRC histopathology. Ethical considerations were prioritized, with patient data anonymized to protect confidentiality and
adhere to consent requirements, while access to the dataset was restricted to anonymized forms for research purposes.
The dataset, which supports further analysis, is available at https://zenodo.org/records/53169

> Kather JN, Weis CA, Bianconi F, Melchers SM, Schad LR, Gaiser T, Marx A, Zollner F: Multi-class texture analysis in
> colorectal cancer histology (2016), Scientific Reports (in press)

## Copyright and License

All rights reserved. No portion of this series may be reproduced, distributed, or transmitted in any form or by any
means, such as photocopying, recording, or other electronic or mechanical methods, without the express written consent
of the author. Exceptions are made for brief quotations included in critical reviews and certain other noncommercial
uses allowed under copyright law. For inquiries regarding permission, please contact the author directly. 
