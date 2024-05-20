from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, accuracy_score, recall_score

# 这部分可以放到utils里
def cal_metrics(true_labels, predicted_labels, prob_labels):
    # 计算 AUC
    auc = roc_auc_score(true_labels, prob_labels[:, 1])
    # 计算准确率（Accuracy）
    accuracy = accuracy_score(true_labels, predicted_labels)
    # 计算召回率（Recall）
    recall = recall_score(true_labels, predicted_labels, average='macro')
    # 计算特异性（Specificity）
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    specificity = tn / (tn + fp)
    # 计算指标f1_score
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    return auc, accuracy, f1, recall, specificity
