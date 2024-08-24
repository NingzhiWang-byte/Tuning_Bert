import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

# Load BERT tokenizer
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('./model_save', do_lower_case=True)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained(
    "./model_save",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
)

# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, delimiter=',')
    n = int(len(df) / 4)
    df = df[n*3:].copy()
    df = df.dropna()
    
    df = df.rename(columns={'code_year': 'file_name'})
    df = df[['file_name', 'sentence', 'class']]
    df['label'] = np.random.randint(0, 2, size=len(df))
    
    print(f'Number of test sentences: {df.shape[0]:,}\n')
    return df

df = load_and_preprocess_data("./new_all_text_labels_0525.csv")

# Preprocess filenames
def preprocess_filename(filename):
    filename = filename.replace('_', '')
    try:
        return int(filename)
    except ValueError:
        return filename

processed_filenames = [preprocess_filename(fname) for fname in df.file_name.values]

# Tokenize sentences
def tokenize_sentences(sentences, labels, max_length=250):
    input_ids = []
    attention_masks = []
    
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'].squeeze())
        attention_masks.append(encoded_dict['attention_mask'].squeeze())
    
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels)
    
    return input_ids, attention_masks, labels

input_ids, attention_masks, labels = tokenize_sentences(df.sentence.values, df.label.values)

# Create DataLoader
def create_dataloader(input_ids, attention_masks, labels, filename_indices, batch_size=32):
    dataset = TensorDataset(input_ids, attention_masks, labels, filename_indices)
    sampler = SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)

filename_indices = torch.arange(len(processed_filenames))
prediction_dataloader = create_dataloader(input_ids, attention_masks, labels, filename_indices)

# Save DataLoader and DataFrame
torch.save(prediction_dataloader, './prediction_dataloader4.pt')
df.to_csv('df_labels.csv', index=False)

print("DataLoader and DataFrame saved successfully.")