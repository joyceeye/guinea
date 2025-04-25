import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader

class ArxivDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

def load_data(data_path):
    df = pd.read_csv(data_path)
    
    df['text'] = df['titles'] + " " + df['summaries']

    # safely convert string representation of lists to actual lists
    df['labels'] = df['terms'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    # filter to only include standard arXiv labels (cs.*, math.*, stat.*)
    def filter_labels(label_list):
        return [label for label in label_list if label.startswith(('cs.', 'math.', 'stat.'))]
    
    # check if all labels are valid (cs.*, math.*, stat.*)
    def all_labels_valid(label_list):
        return all(label.startswith(('cs.', 'math.', 'stat.')) for label in label_list)
    
    # keep only rows where ALL labels are valid
    df = df[df['labels'].apply(all_labels_valid)]
    
    # apply the filter to ensure we only keep the valid labels
    df['labels'] = df['labels'].apply(filter_labels)
    
    print(f"filtered dataset: {len(df)} rows")
    
    # inspect the unique labels
    unique_labels = set()
    for labels in df['labels']:
        unique_labels.update(labels)
    print(f"found {len(unique_labels)} unique labels: {unique_labels}")
    
    return df

def prepare_data(df, tokenizer, max_length, batch_size, mlb=None):
    # fit multilabel binarizer if not provided
    if mlb is None:
        mlb = MultiLabelBinarizer()
        label_matrix = mlb.fit_transform(df['labels'])
    else:
        # use existing mlb for consistent label dimensions
        label_matrix = mlb.transform(df['labels'])
    
    # create dataset
    dataset = ArxivDataset(
        texts=df['text'].tolist(),
        labels=label_matrix,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return dataloader, mlb