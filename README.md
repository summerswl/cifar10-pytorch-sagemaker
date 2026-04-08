# CIFAR-10 Image Classifier with PyTorch & AWS SageMaker

A complete end-to-end deep learning project that trains a Convolutional Neural Network (CNN) to classify 32×32 color images into 10 categories and deploys it as a live REST API on AWS SageMaker.

## Project Overview

This project demonstrates the full machine learning workflow:
- Training a CNN on the CIFAR-10 dataset
- Using data augmentation for better generalization
- Packaging and deploying the model to AWS SageMaker
- Making real-time predictions via API

**Final Test Accuracy**: ~84% (on CPU training)

## Classes

The model classifies images into one of the following 10 categories:
- **airplane**, **automobile**, **bird**, **cat**, **deer**, **dog**, **frog**, **horse**, **ship**, **truck**

## Project Structure

cifar10-pytorch-sagemaker/
├── train.py                 # Trains the CNN with data augmentation
├── inference.py             # SageMaker inference handler
├── deploy.py                # Deploys model to SageMaker
├── test_endpoint.py         # Tests the live endpoint
├── app.py                   # Gradio web interface (optional)
├── requirements.txt         # Python dependencies
├── model.pth                # Trained model weights
├── model.tar.gz             # Packaged model for SageMaker
└── README.md

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
python deploy.py
python app.py #deploys app locally to http://127.0.0.1:7860
