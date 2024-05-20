import os
import sys
import h5py
import random
import pandas as pd
sys.path.append('../')
import config_h.configs_liver as configs 
from torch.utils.data import Dataset, DataLoader

class LiverDataset(Dataset):
    def __init__(self, config, transform=None):
        self.config = config
        if self.config.mode == 'train':
            self.meta_data = pd.read_excel(config.train_data_file)
        else:
            self.meta_data = pd.read_excel(config.test_data_file)
        self.meta_data = self.meta_data[self.meta_data[self.config.task].isin(self.config.label_dict[self.config.task].keys())]
        self.transform = transform
        self.label_dict = self.config.label_dict # {"Non-desmoplastic": 0, "desmoplastic": 1}

    def __len__(self):
        # 返回数据集的大小
        return len(self.meta_data)

    def __getitem__(self, idx):
        # 读取idx对应的数据和标签
        coad_files = eval(self.meta_data.iloc[idx]['肠癌特征路径'])
        liver_files = eval(self.meta_data.iloc[idx]['肝转移特征路径'])

        coad_files = [i for i in coad_files if os.path.exists(i)]
        liver_files = [i for i in liver_files if os.path.exists(i)]

        all_files = coad_files + liver_files

        if len(all_files) == 0:
            raise Excpetion("No file to read !!!!")
        else:
            # 从coad files 中随机选择一个文件
            all_files = random.choice(all_files)
            file_id = os.path.basename(all_files)
            # 读取数据
            with h5py.File(all_files, 'r') as f:
                img_data = f['features'][:]

        # transfer label
        label = self.meta_data.iloc[idx][self.config.task]
        # print('label: ', label)

        label = self.label_dict[self.config.task][label]

        # 数据预处理
        if self.transform:
            img_data = self.transform(img_data)


        # 返回数据和标签
        return file_id, img_data, label

if __name__ == '__main__':
    # 测试代码
    # 创建数据集实例
    CONFIGS = {
        'Liver_WIKG': configs.get_Liver_wikg_config(),
    }
    config = CONFIGS['Liver_WIKG']
    dataset = LiverDataset(config)
    # 打印数据集大小
    print(len(dataset))
    # 随机选择一个样本
    sample = random.choice(dataset)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        print(batch[0], batch[1].shape, batch[2].shape)
        print(type(batch[0]))
        break


