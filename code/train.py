import os
import torch
import numpy as np
from utils.utils import cal_metrics
from models.transmil import TransMIL
from models.simple_model import MIL_fc
from torch.utils.data import DataLoader
import config_h.configs_liver as configs 
from datasets_h.dataset_feature import LiverDataset
def main():
    # 加载超参配置
    CONFIGS = {
        'Liver_WIKG': configs.get_Liver_wikg_config(),
    }
    config = CONFIGS['Liver_WIKG']
    # 加载数据集
    batch_size = config.batch_size

    config.mode = "train"
    train_dataset = LiverDataset(config)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    config.mode = "test"
    test_dataset = LiverDataset(config)
    # test_dataset = test_dataset[:10]
    # 取test_dataset的前十个用于测试
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 打印训练数据集和测试集的数目
    print("训练数据集的数目:", len(train_dataset))
    print("测试数据集的数目:", len(test_dataset))

    # 初始化模型
    model = TransMIL(config.n_classes).cuda()

    # 初始化损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # model = WiKG(dim_in=config.dim_in, dim_hidden=config.dim_hidden, topk=config.topk, n_classes=config.n_classes, agg_type=config.agg_type, dropout=config.dropout, pool=config.pool).cuda()
    # 训练模型
    model.train()
    for epoch in range(config.eps):
        predicted_labels = []
        true_labels = []
        prob_labels = []
        for i, batch in enumerate(train_dataloader):
            f_id, img_f, label = batch
            # print(type(img_f), img_f.shape)
            # zero
            optimizer.zero_grad()
            # 前向传播
            # print(img_f.cuda().shape)
            results_dict = model(data = img_f.cuda())
            # print(results_dict)
            outputs = results_dict['logits']
            probs = results_dict['Y_prob']
            predicted = results_dict['Y_hat']
            # 计算损失
            loss = criterion(outputs, label.cuda())
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            # 训练模型每个epoch评价指标打印
            predicted_labels.append(predicted.cpu().numpy())
            true_labels.append(label.cpu().numpy())
            prob_labels.append(probs.cpu().detach().numpy())

            # 打印训练信息
            if ((i+1) % 5) == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, config.eps, i+1, len(train_dataloader), loss.item()))

        predicted_labels = np.concatenate(predicted_labels, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)
        prob_labels = np.concatenate(prob_labels, axis=0)


        auc, accuracy, f1, recall, specificity = cal_metrics(true_labels, predicted_labels, prob_labels)
        print('【TRAIN】 Epoch [{}/{}], roc_auc: {:.4f}, accuracy: {:.4f}, f1_score: {:.4f}, recall: {:.4f}, specificity: {:.4f}'.format(epoch+1, config.eps, auc, accuracy, f1, recall, specificity))


        # break
        # 内部验证模型
        model.eval()
        predicted_labels = []
        true_labels = []
        prob_labels = []
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in test_dataloader:
                f_id, img_f, label = batch
                results_dict = model(data=img_f.cuda())
                outputs = results_dict['logits']
                probs = results_dict['Y_prob']
                predicted = results_dict['Y_hat']

                predicted_labels.append(predicted.cpu().numpy())
                true_labels.append(label.cpu().numpy())
                prob_labels.append(probs.cpu().numpy())


            predicted_labels = np.concatenate(predicted_labels, axis=0)
            true_labels = np.concatenate(true_labels, axis=0)
            prob_labels = np.concatenate(prob_labels, axis=0)

            # 计算 AUC
            auc, accuracy, f1, recall, specificity = cal_metrics(true_labels, predicted_labels, prob_labels)
            print('【VALID】 Epoch [{}/{}], roc_auc: {:.4f}, accuracy: {:.4f}, f1_score: {:.4f}, recall: {:.4f}, specificity: {:.4f}'.format(epoch+1, config.eps, auc, accuracy, f1, recall, specificity))
            
            # 保存评价指标到csv文件中
            with open(config.evaluation_save_path, 'a') as f:
                # 起个列名
                if epoch == 0:
                    f.write('valid epoch,roc_auc,accuracy,f1_score,sensitivity,specificity\n')
                f.write('{},{},{},{}\n'.format(epoch+1, auc, accuracy, f1, recall, specificity))
            # 保存模型参数
            torch.save(model.state_dict(), config.model_save_path + f"epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()