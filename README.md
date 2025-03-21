# Food101-classifier


# Food101 Image Classification with TensorFlow

This repository contains a TensorFlow-based deep learning model for classifying images from the Food101 dataset. The project leverages EfficientNetB0 as a feature extractor and fine-tunes the model to achieve better classification performance.

## Project Overview

- Used TensorFlow Datasets to load and preprocess the Food101 dataset.
- Implemented EfficientNetB0 as a feature extractor and fine-tuned it for improved accuracy.
- Modularized the codebase with separate scripts for data loading, feature extraction, and fine-tuning.
- Currently, the repository includes only the core scripts, with plans to add more explanatory content.

## Updates
- **[2025-03-04]** - Initial version of the project has been released.
- **[2025-03-21]** - The project is being updated with benchmark improvements.

## Repository Structure

```bash
datasets/
├── load_data.py              # Loads and preprocesses the Food101 dataset
models/
├── feature_extractor.py      # Initializes EfficientNetB0 as a feature extractor
├── fine_tune.py              # Fine-tunes the model for improved accuracy
helper_functions/
├── helper_functions.py       # Contains utility functions for flexible results

