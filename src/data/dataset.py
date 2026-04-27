import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os

def remove_stopwords(text, stop_words):
    if not stop_words:
        return text
    words = str(text).split(' ')
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

class NewsDataset(Dataset):
    def __init__(self, X, y, stop_words=None):
        super().__init__()
        self.X = X
        self.y = y
        self.stop_words = stop_words
        
        print("Cleaning text...")
        self.X_cleaned = [remove_stopwords(content, self.stop_words) for content in tqdm(X)]
        self.tokenized_X = None

    def tokenize(self, tokenizer, max_length=8000):
        print("Tokenizing data...")
        self.tokenized_X = tokenizer(self.X_cleaned,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=max_length,
                                    return_tensors="pt"
                                )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.tokenized_X is None:
            return self.X_cleaned[idx], self.y[idx]
        
        item = {key: val[idx] for key, val in self.tokenized_X.items()}
        # The notebook seems to return (input_ids, attention_mask, label)
        # but let's see how it's used in training
        return item['input_ids'], item['attention_mask'], torch.tensor(self.y[idx])

def load_stopwords(filepath):
    if not os.path.exists(filepath):
        print(f"Stopwords file not found at {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as file:
        stop_words = file.read().split('\n')
    return stop_words
