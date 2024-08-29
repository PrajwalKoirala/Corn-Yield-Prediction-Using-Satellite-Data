# Corn-Yield-Prediction-Using-Satellite-Data

## Overview

This project uses a combination of VisualBERT and ResNet models to predict corn yield from satellite data. We finetune a pretrained VisualBERT model (`uclanlp/visualbert-vqa-coco-pre`) and ResNet model to encode the relevant information for each data point. A pretrained BERT tokenizer (`google-bert/bert-base-uncased`) is used for tokenizing non-image features.

## Model Approach

- **VisualBERT**: Combines information from text and image for robust feature extraction. Handles a variable number of satellite images.
- **ResNet**: Processes satellite images to capture spatial features.
- **FCNN**: Fully Connected Neural Network used for the final yield prediction.
- **Ensemble**: An ensemble of four models is used, with the median prediction providing the final output.

## Usage

- **Training**: Use `training/training.ipynb` to retrain the models. Model definitions are in `training/model.py`.
- **Inference**: Run `training/inference_ensemble.ipynb` to reproduce the inference results.

## Pretrained Models

Download pretrained models for inference from this [link](https://iastate.box.com/s/pjjklumriagsgf1s08skf11htfz54a52).

## Requirements

- Python 3.8+, PyTorch, Transformers, OpenCV, scikit-learn, pandas, numpy, Jupyter Notebook.

