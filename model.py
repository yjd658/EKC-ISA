import torch.nn as nn
from transformers import BertModel,RobertaTokenizer, RobertaModel
from transformers import BertTokenizer,AutoModelForSequenceClassification,AutoTokenizer
from transformers.adapters import BertAdapterModel, AutoAdapterModel,AdapterConfig,CompacterConfig,ConfigUnion, ParallelConfig, PrefixTuningConfig,UniPELTConfig,LoRAConfig,IA3Config,Parallel, MAMConfig,UniPELTConfig
import torch
import transformers.adapters.composition as ac
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,classification_report
import torch.nn.functional as F
criterion = nn.CrossEntropyLoss()
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def report_metrics(total_labels, total_preds):

    class_names = ['Neutral', 'Positive', 'Negative']


    precision = precision_score(total_labels, total_preds, average='macro')
    recall = recall_score(total_labels, total_preds, average='macro')
    f1 = f1_score(total_labels, total_preds, average='macro')
    accuracy = accuracy_score(total_labels, total_preds)


    report = classification_report(total_labels, total_preds, target_names=class_names,digits=4)


    print("Macro Precision:", precision)
    print("Macro Recall:", recall)
    print("Macro F1:", f1)
    print("Accuracy:", accuracy)
    print(report)

def visualize_feature_space(features, labels, epoch, save_folder="./d_w"):

    tsne = TSNE(n_components=2)
    features_2d = tsne.fit_transform(features)


    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, s=5)


    plt.title("Feature Space Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")


    current_dir = os.getcwd()


    save_folder_path = os.path.join(current_dir, save_folder)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)


    save_path = os.path.join(save_folder_path, f"feature_space_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=100)  # 设置dpi为100，相当于每英寸100个像素


    plt.show()

def similarity_loss3(features, labels,weight,N,total_class_sums):
    num_classes = 3


    class_sums = torch.zeros(num_classes, features.size(-1), device=features.device)
    class_means = torch.zeros(num_classes, features.size(-1), device=features.device)
    class_counts = torch.zeros(num_classes, device=features.device)

    for i in range(num_classes):
        class_features = (features[labels == i])
        features_weight=(weight[labels == i])
        class_counts[i] = class_features.size(0)
        if class_counts[i] == 0:
            print("此batch中，第{}类数据不存在".format(i))
            continue

        class_sums[i]=torch.mean(class_features,dim=0)

    class_means=class_sums


    loss=0
    for m in range(num_classes):
        if class_counts[m] == 0:
            continue
        class_features = features[labels == m]

        cosine_diff=torch.zeros(class_features.shape[0], device=features.device)
        cosine_similarities=torch.zeros(class_features.shape[0], device=features.device)
        for j in range(num_classes):
            if class_counts[m] == 0:
                continue
            if (j!=m):
                other_mean=class_means[j].expand(class_features.shape[0], -1)
                cosine_diff = cosine_diff+torch.exp( F.cosine_similarity(class_features, other_mean)/0.05)


        expanded_mean = class_means[m].expand(class_features.shape[0], -1)
        cosine_similarities =torch.exp( F.cosine_similarity(class_features, expanded_mean)/0.05)
        cosine_diff=cosine_diff+cosine_similarities

        loss +=(-torch.log(cosine_similarities/cosine_diff)).sum()

    a=0
    for k in range(num_classes):
        for l in range(k + 1, num_classes):
            if class_counts[k] == 0 or class_counts[l] == 0:
                continue
            a +=-torch.log(1/(1+torch.exp(F.cosine_similarity(class_means[k].unsqueeze(0),class_means[l].unsqueeze(0))/0.05)))
    a=a/2
    loss = loss+a
    return loss,total_class_sums.detach()

class SentimentClassifier_v8(nn.Module):
    def __init__(self, bert_model_name=r'hfl/chinese-roberta-wwm-ext', num_labels=3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)


        self.max_seq_length=200

        self.adapterconfig=LoRAConfig(r=4, alpha=16)
        self.bert.add_adapter("main_adapter", config=self.adapterconfig)
        self.bert.set_active_adapters("main_adapter")
        self.bert.train_adapter("main_adapter")


        self.mlp = nn.Sequential(
            nn.Linear(256 *3, 3),


        )


        self.num_labels = num_labels
        dropout_prob = 0.5
        self.dropout = nn.Dropout(dropout_prob)

        self.lstm = nn.LSTM(300, 384, 1, batch_first=True, bidirectional=True)
        # self.fc = nn.Linear(hidden_size, num_labels)
        # self.U = torch.nn.Parameter(torch.randn(768, 3, 768))
        self.multiattn = nn.MultiheadAttention(768, 3)
        self.wt = nn.Sequential(
            nn.Linear(1, 6, bias=False),
            nn.ReLU(),
            nn.Linear(6, 1, bias=False)
        )



    def forward(self, input_ids, attention_mask,commonsense,labels):
        # input_embeddings = self.bert.embeddings.word_embeddings(input_ids)
        input_ids = input_ids [:, :self.max_seq_length]
        attention_mask = attention_mask[:, :self.max_seq_length]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # outputs1 = self.bert2(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        lstm_output, _ = self.lstm(commonsense)

        query_norm = pooled_output.unsqueeze(1)
        key_norm = F.layer_norm(lstm_output.transpose(0, 1), normalized_shape=lstm_output.transpose(0, 1).size()[1:])
        value_norm = F.layer_norm(lstm_output.transpose(0, 1), normalized_shape=lstm_output.transpose(0, 1).size()[1:])
        query_norm=query_norm.permute(1, 0, 2)


        attn_output, attn_output_weights = self.multiattn(query_norm, key_norm, value_norm)

        out=attn_output.squeeze()
        out=F.normalize(out, p=2, dim=1)


        conbintation=(query_norm+out).squeeze()

        logits =F.softmax( self.mlp(conbintation),dim=1)

        diff=F.cross_entropy(logits, labels, reduction='none')
        diff=(1/diff).unsqueeze(1)

        loss1 = criterion(logits, labels)
        weight =self.wt(  diff.to('cuda'))

        return logits, conbintation, loss1, weight



class en_SentimentClassifier(nn.Module):
    def __init__(self, bert_model_name='roberta-base', num_labels=3):
        super().__init__()

        self.bert =  BertModel.from_pretrained('bert-base-uncased')


        self.max_seq_length = 100

        self.adapterconfig = LoRAConfig(r=8, alpha=16)
        self.bert.add_adapter("main_adapter", config=self.adapterconfig)
        self.bert.set_active_adapters("main_adapter")
        self.bert.train_adapter("main_adapter")


        self.mlp = nn.Sequential(
            nn.Linear(256 * 3, 3),

        )




        self.num_labels = num_labels
        dropout_prob = 0.5
        self.dropout = nn.Dropout(dropout_prob)

        self.lstm = nn.LSTM(300, 384, 1, batch_first=True, bidirectional=True)

        self.multiattn = nn.MultiheadAttention(768, 3)
        self.wt = nn.Sequential(
            nn.Linear(1, 6, bias=False),
            nn.ReLU(),
            nn.Linear(6, 1, bias=False)
        )

    def forward(self, input_ids, attention_mask, commonsense, labels):

        input_ids = input_ids[:, :self.max_seq_length]
        attention_mask = attention_mask[:, :self.max_seq_length]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)


        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        lstm_output, _ = self.lstm(commonsense)

        query_norm = pooled_output.unsqueeze(1)
        key_norm = F.layer_norm(lstm_output.transpose(0, 1), normalized_shape=lstm_output.transpose(0, 1).size()[1:])
        value_norm = F.layer_norm(lstm_output.transpose(0, 1), normalized_shape=lstm_output.transpose(0, 1).size()[1:])
        query_norm = query_norm.permute(1, 0, 2)

        attn_output, attn_output_weights = self.multiattn(query_norm, key_norm, value_norm)

        out = attn_output.squeeze()
        out = F.normalize(out, p=2, dim=1)

        conbintation = (query_norm + out).squeeze()

        logits = F.softmax(self.mlp(conbintation), dim=1)

        diff = F.cross_entropy(logits, labels, reduction='none')
        diff = (1 / diff).unsqueeze(1)

        loss1 = criterion(logits, labels)
        weight = self.wt(diff.to('cuda'))

        return logits, conbintation, loss1, weight








