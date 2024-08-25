# Chinese BERT Text Classification

This project implements a text classification model using BERT (Bidirectional Encoder Representations from Transformers) for Chinese language texts. The model is trained to perform binary classification on sentences.

## Features

- Utilizes the `bert-base-chinese` pre-trained model
- Implements data preprocessing and tokenization
- Performs model fine-tuning for binary classification
- Includes training and validation loops
- Saves the trained model for future use

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- Pandas
- NumPy
- Google Colab (optional, for GPU acceleration)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/chinese-bert-classification.git
   cd chinese-bert-classification
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your data:
   - Place your Excel file (`rand_.xlsx`) in the project directory
   - Ensure it contains 'sentence' and 'label' columns

2. Run the script:
   ```
   python main.py
   ```

3. The script will:
   - Load and preprocess the data
   - Tokenize the sentences
   - Split the data into training and validation sets
   - Fine-tune the BERT model
   - Evaluate the model's performance
   - Save the trained model

## Model Architecture

- Base Model: `bert-base-chinese`
- Classification Layer: Linear layer with 2 output neurons (binary classification)

## Training Details

- Epochs: 4
- Batch Size: 32
- Learning Rate: 2e-5
- Optimizer: AdamW
- Loss Function: Cross-Entropy Loss

## Output

The trained model and tokenizer will be saved in the `./model_save/` directory.



## Machine Learning

This repository contains Matlab code for ensemble learning algorithms aimed at predicting greenwashing.

### Main Files

- **`run_RUSBoost.m`**: This is the Matlab code to run the RUSBoost ensemble learning algorithm for predicting greenwashing. 
- **`data_reader.m`**: Required for reading the data necessary for the model.
- **`evaluate.m`**: Used for evaluating model performance.

### Additional Algorithms

We also provide training code for other algorithms, including:
- Random Forest
- LSTM
- SVM
- And others

### Hyper-Parameter Tuning

- **`tune_RUSBoost.m`**: This Matlab code replicates the hyper-parameter tuning for our RUSBoost model. The number of learners/trees is tuned using a traditional grid search approach. The parameter space is manually specified as:
[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000]

For each hyper-parameter, the model is trained using the training period from 2008 to 2018 and evaluated in terms of AUC during the validating years 2019 to 2022.


## Acknowledgments

- Hugging Face Transformers library
- BERT paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
