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

## Acknowledgments

- Hugging Face Transformers library
- BERT paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
