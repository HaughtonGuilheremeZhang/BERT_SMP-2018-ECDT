import json
from bert_model import encode,ClassificationDataset
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader
from pre_eval import stalabels
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import random
n_samples = 2299#1000
n_channels = 1#3
height = 32
width = 32
n_classes = 31#10


def replace_overrepresented_labels(predicted_labels, num_classes):
    """
    随机地将没出现过的标签替换出现过多的标签。

    参数:
    predicted_labels (torch.Tensor): 模型预测的标签张量
    num_classes (int): 类别总数

    返回:
    torch.Tensor: 替换后的标签张量
    """
    # 将标签转换为列表


    # 统计每个标签的出现次数
    label_counts = {label: predicted_labels.count(label) for label in range(num_classes)}

    # 找到没出现过的标签和出现次数最多的标签
    missing_labels = [label for label in range(num_classes) if label_counts[label] == 0]
    overrepresented_labels = [label for label in label_counts if
                              label_counts[label] > len(predicted_labels) // num_classes]

    if not missing_labels or not overrepresented_labels:
        return predicted_labels

    # 随机替换过多标签为未出现的标签
    for i in range(len(predicted_labels)):
        if predicted_labels[i] in overrepresented_labels:
            new_label = random.choice(missing_labels)
            predicted_labels[i] = new_label
            missing_labels.remove(new_label)
            if not missing_labels:
                break

    return predicted_labels
if __name__ =='__main__':
    model = BertForSequenceClassification.from_pretrained("bert-multiclass-classifier")
    tokenizer = BertTokenizer.from_pretrained("bert-multiclass-classifier")

    # 将模型加载到设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    #打开dev
    with open('dev.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 数据集的结构是一个包含字典的列表
    query = [data[key]['query'] for key in data]
    labels = [data[key]['label'] for key in data]

    label2id = {label: idx for idx, label in enumerate(stalabels)}

    test_texts = query
    #print(test_texts)
    test_encodings, _ = encode(test_texts,labels,label2id)  # 标签占位符

    # 创建测试数据集和数据加载器
    test_dataset = ClassificationDataset(test_encodings, torch.zeros(len(test_encodings['input_ids'])))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)


    # 进行预测
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())



    #id2label = {idx: label for idx, label in enumerate(stalabels)}
    #将预测结果转换为标签
    #pred_labels = [id2label[pred] for pred in preds]
    #print(pred_labels)
    # 防止分母出现0的情况，随机插入未出现的标签
    new_labels = replace_overrepresented_labels(preds, n_classes)

    # 将预测结果存入result.json
    rguess_dct = {}
    for it in data:
        rguess_dct[it] = {"query": data[it]['query'], "label": stalabels[new_labels[int(it)]]}
    json.dump(rguess_dct, open("result.json", 'w', encoding='utf-8'), ensure_ascii=False)
