import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from data_process import read_data,SentimentDataset

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,classification_report
import torch.nn as nn
from model import en_SentimentClassifier,report_metrics,visualize_feature_space,similarity_loss3
from parameters import disable_parameters, print_params_stat
from torch.utils.data import WeightedRandomSampler

import random
import torch
import time
torch.cuda.empty_cache()
train_data = read_data('E_train.txt','E_train.pt',lanaugue='en')
dev_data = read_data('E_dev.txt','E_dev.pt',lanaugue='en')
test_data = read_data('E_test.txt','E_test.pt',lanaugue='en')

train_dataset = SentimentDataset(train_data)
dev_dataset = SentimentDataset(dev_data)
test_dataset = SentimentDataset(test_data)


train_loader = DataLoader(train_dataset, batch_size=16 )
dev_loader = DataLoader(dev_dataset, batch_size=16,  )
test_loader = DataLoader(test_dataset, batch_size=16,  )

device = torch.device('cuda:0')
model = en_SentimentClassifier().to(device)

for name, param in model.named_parameters():
    print(f'Parameter: {name}, Requires Gradient: {param.requires_grad}')




optimizer =torch.optim.AdamW(model.parameters(), lr=1e-4)
num_epochs = 50
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
criterion = nn.CrossEntropyLoss()
# print_params_stat(model)
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_preds = []
    total_labels = []
    with torch.no_grad():
        total_class_sums = torch.zeros(3, 768, device=device)
        total_class_counts = torch.zeros(3, device=device)
        start_time = time.time()
        for batch in dataloader:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            labels = batch['labels'].squeeze().to(device)
            commonsense = batch['commonsense'].to(device)
            outputs, conbintation, loss1, weight = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                commonsense=commonsense,
                labels=labels,
            )


            loss3, total_class_counts = similarity_loss3(conbintation, labels, weight, N, total_class_sums)
            H=loss1/loss3
            loss = (0.7) * loss1 + 0.3 *H* loss3
            total_loss += loss.item()
            logits = outputs
            preds = torch.argmax(logits, dim=1)
            total_preds.extend(preds.tolist())
            total_labels.extend(labels.tolist())
        end_time = time.time()
        reference_time = (end_time - start_time)/len(dataloader)
        print("reference_time:",{reference_time})
    avg_loss = total_loss / len(dataloader)
    report_metrics(total_labels, total_preds)
    f1 = f1_score(total_labels, total_preds, average='macro')
    return avg_loss, f1



for epoch in range(num_epochs):
    model.train()
    print_params_stat(model)
    for name, para in model.named_parameters():
        model.state_dict()[name][:] += (torch.rand(para.size(), device=device) - 0.5) * 0.2 * torch.std(para)
    total_loss = 0.0
    total_preds = []
    total_labels = []
    total_class_sums = torch.zeros(3, 768, device=device)
    total_class_counts = torch.zeros(3, device=device)
    t_p = torch.tensor([]).to(device)
    t_l = torch.tensor([]).to(device)
    start_time = time.time()
    N = 0
    for batch in train_loader:
        N=N+1
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].squeeze().to(device)
        commonsense=batch['commonsense'].to(device)
        optimizer.zero_grad()

        outputs, conbintation, loss1, weight = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            commonsense=commonsense,
            labels=labels,
        )

        # loss1 =criterion(outputs, labels)
        # weight=1/loss1
        loss3, total_class_counts = similarity_loss3(conbintation, labels, weight, N, total_class_sums)
        H = loss1 / (loss3)
        loss=0.7*loss1+0.3*H*loss3
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        logits = outputs
        preds = torch.argmax(logits, dim=1)
        total_preds.extend(preds.tolist())
        total_labels.extend(labels.tolist())
        t_p = torch.cat((t_p, outputs), dim=0)
        t_l = torch.cat((t_l, labels), dim=-1)
    end_time = time.time()  # 记录当前时间
    epoch_time = end_time - start_time
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Time: {epoch_time:.2f} seconds")
    avg_loss = total_loss / len(train_loader)
    report_metrics(total_labels, total_preds)
    train_f1 = f1_score(total_labels, total_preds, average='macro')
    print(f"Epoch {epoch+1}/{num_epochs}, loss: {avg_loss:.4f}",f"Train F1 score: {train_f1:.4f}")
    val_loss, val_f1_score = evaluate(model, dev_loader)
    print(f"Validation loss: {val_loss:.4f}, F1 score: {val_f1_score:.4f}")
    test_loss, test_f1_score = evaluate(model, test_loader)
    print(f"Test loss: {test_loss:.4f}, F1 score: {test_f1_score:.4f}")
    if epoch % 1 == 0:
        visualize_feature_space(t_p.cpu().detach().numpy(), t_l.cpu().detach().numpy(),epoch)
