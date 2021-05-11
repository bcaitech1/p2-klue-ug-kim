import pickle
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class RE_Dataset(Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 불러온 tsv 파일을 원하는 형태의 df로 변경
def preprocessing_dataset(dataset, label_type):
    label = []
    for i in dataset[8]:
        if i == "blind":  # test datsaet의 경우
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame({'sentence':dataset[1], 'entity_01':dataset[2], 'entity_02':dataset[5], 'label':label,})
    return out_dataset


def load_data(dataset_dir):
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)  # dictionary
    
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    dataset = preprocessing_dataset(dataset, label_type)

    return dataset

# basic input tokenizing
# 다양한 종류의 tokenizer와 special token 활용해보기
def tokenized_dataset(dataset, tokenizer):
    concat_entity = []
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        # temp = e01 + '</s></s>' + e02  # roberta
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        add_special_tokens=True,
    )

    return tokenized_sentences


class TokenDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tokenized_datasets = self.tokenized_dataset(self.dataset, self.tokenizer)
        item = {key: val[idx].clone().detach() for key, val in tokenized_datasets.items()}
        item['labels'] = torch.tensor(list(self.dataset['label'])[idx])
        return item

    def __len__(self):
        return len(self.dataset)
        
    def tokenized_dataset(self, dataset, tokenizer):
        concat_entity = []
        for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
            temp = ''
            temp = e01 + '[SEP]' + e02
            # temp = e01 + '</s></s>' + e02  # roberta
            concat_entity.append(temp)
        
        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset['sentence']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=True,
        )

        return tokenized_sentences
