# Food101-classifier


This repository contains a TensorFlow-based deep learning model for classifying images from the Food101 dataset. The project leverages EfficientNetB0 as a feature extractor and fine-tunes the model to achieve better classification performance.

 Project Overview
   Used TensorFlow Datasets to load and preprocess the Food101 dataset.
   Implemented EfficientNetB0 as a feature extractor and fine-tuned it for improved accuracy.
   Modularized the codebase with separate scripts for data loading, feature extraction, and fine-tuning.
   Currently, the repository includes only the core scripts, with plans to add more explanatory content.

Repository Structure
  load_data.py → Loads and preprocesses the Food101 dataset.
  create_feature_extractor.py → Initializes EfficientNetB0 as a feature extractor.
  fine_tune.py → Fine-tunes the model for improved accuracy.
  requirements.txt → Lists the necessary dependencies.
  [Upcoming] A Jupyter/Colab Notebook with detailed explanations and training results.

Next Steps
  Add an explanatory notebook with step-by-step details.
  Include evaluation metrics & training results.
  Optimize hyperparameters and address any potential issues.
