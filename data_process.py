import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn

device = torch.device('cuda:0')
def read_data(file_path,commonsense,lanaugue=None):
    data = []
    commonsense_embedding=torch.load(commonsense)
    if (lanaugue=='en'):
        tokenizer = BertTokenizer.from_pretrained(r'bert-base-uncased')
        with open(file_path, 'r', encoding='utf-8') as f:
            for i,line in enumerate(f):
                parts = line.strip().split()  # Split line by spaces
                text = ' '.join(parts[:-1])  # Join all parts except the last one as text
                label = parts[-1]  # The last part is the label
                tokenized_text = tokenizer.encode(text, max_length=512, truncation=True, padding='max_length')
                data.append((torch.tensor(tokenized_text), int(label),commonsense_embedding[i]))
    else:
        tokenizer = BertTokenizer.from_pretrained(r'bert-base-chinese')
        with open(file_path, 'r', encoding='utf-8') as f:
            for i,line in enumerate(f):
                text, label = line.strip().split()
                tokenized_text = tokenizer.encode(text, max_length=512, truncation=True, padding='max_length')
                data.append((torch.tensor(tokenized_text), int(label),commonsense_embedding[i]))
    return data



class SentimentDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        input_ids, label,commonsense = self.data[index]
        return {
            'input_ids': input_ids,
            'attention_mask': (input_ids != 0).long(),
            'labels': label,
            'commonsense':commonsense
        }

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        # 获取所有样本的标签列表
        labels = [sample[1] for sample in self.data]  # 替换为实际的标签字段名称
        return labels

    def get_class_weights(self):
        # 计算每个类别的样本数量
        labels = self.get_labels()
        labels_tensor = torch.tensor(labels)
        class_counts = torch.bincount(labels_tensor)

        # 计算每个类别的权重
        class_weights = 1.0 / class_counts
        return class_weights


