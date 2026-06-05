# ChestX-ray14 数据集加载器，支持多标签分类（14种疾病）
# 数据集来源：NIH Chest X-ray Dataset (Wang et al., CVPR 2017)

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

CHESTXRAY14_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]


class ChestXray14(Dataset):
    """NIH ChestX-ray14 多标签数据集

    每张X光片可能包含0~14种疾病标签，"No Finding"表示所有标签为0。
    图像为1024x1024灰度PNG，自动缩放到指定尺寸并转为3通道。
    """

    def __init__(self, data_dir, transform=None, image_size=224):
        csv_path = os.path.join(data_dir, 'Data_Entry_2017.csv')
        df = pd.read_csv(csv_path)

        self.image_dir = os.path.join(data_dir, 'images')
        self.transform = transform
        self.image_size = image_size

        self.filenames = []
        self.labels = []

        for _, row in df.iterrows():
            img_name = row['Image Index']
            finding = row['Finding Labels']

            img_path = os.path.join(self.image_dir, img_name)
            if not os.path.exists(img_path):
                continue

            self.filenames.append(img_path)

            label_vec = np.zeros(len(CHESTXRAY14_LABELS), dtype=np.float32)
            if finding != 'No Finding':
                for disease in finding.split('|'):
                    if disease in CHESTXRAY14_LABELS:
                        label_vec[CHESTXRAY14_LABELS.index(disease)] = 1.0
            self.labels.append(label_vec)

        self.labels = np.array(self.labels)

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.filenames)

    def __getitem__(self, idx):
        """返回指定索引的 (图像, 标签) 元组，图像为 PIL → transform → Tensor"""
        img = Image.open(self.filenames[idx]).convert('L')
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        if self.transform:
            img = self.transform(img)

        label = torch.from_numpy(self.labels[idx])
        return img, label#one-hot 标签向量

    def num_classes(self):
        """返回类别总数（14 种疾病）"""
        return len(CHESTXRAY14_LABELS)

    @staticmethod
    def label_names():
        """返回疾病标签名称列表（14 种胸部X光疾病）"""
        return CHESTXRAY14_LABELS
