import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader

# 标签列表
stalabels = ['website', 'tvchannel', 'lottery', 'chat', 'match', 'datetime', 'weather', 'bus', 'novel', 'video',
             'riddle',
             'calc', 'telephone', 'health', 'contacts', 'epg', 'app', 'music', 'cookbook', 'stock', 'map', 'message',
             'poetry', 'cinemas', 'news', 'flight', 'translation', 'train', 'schedule', 'radio', 'email']


# 创建 PyTorch 数据集
class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


# 数据预处理函数
def encode(texts, labels, label2id, max_length=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    label_ids = [label2id[label] for label in labels]
    return encodings, torch.tensor(label_ids)


# 训练函数
def train(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return total_loss / len(train_loader)


# 测试函数
def evaluate(model, val_loader, device):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return accuracy_score(true_labels, preds)


if __name__ == '__main__':
    # 加载数据集
    with open('train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 数据集的结构是一个包含字典的字典
    query = [data[key]['query'] for key in data]
    labels = [data[key]['label'] for key in data]
    num_epoch=10
    df = pd.DataFrame(data)
    # print(df)
    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        query, labels, test_size=0.1, random_state=42
    )

    # 初始化 BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 标签映射
    label2id = {label: idx for idx, label in enumerate(stalabels)}
    id2label = {idx: label for idx, label in enumerate(stalabels)}

    # 对训练集和验证集进行编码
    train_encodings, train_label_ids = encode(train_texts, train_labels, label2id)
    val_encodings, val_label_ids = encode(val_texts, val_labels, label2id)

    train_dataset = ClassificationDataset(train_encodings, train_label_ids)
    val_dataset = ClassificationDataset(val_encodings, val_label_ids)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=True)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

    from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
    from sklearn.metrics import accuracy_score

    # 初始化 BERT 模型
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=len(labels))

    # 优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 学习率调度器
    total_steps = len(train_loader) * num_epoch  # 训练 num_epoch个 epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 训练和评估
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    for epoch in range(num_epoch):
        train_loss = train(model, train_loader, optimizer, scheduler, device)
        val_accuracy = evaluate(model, val_loader, device)
        print(f'Epoch {epoch + 1}, Loss: {train_loss}, Accuracy: {val_accuracy}')

    # 保存模型
    model.save_pretrained("bert-multiclass-classifier")
    tokenizer.save_pretrained("bert-multiclass-classifier")
    print("success")
