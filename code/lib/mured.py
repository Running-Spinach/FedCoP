# MuReD (Multi-label Retinal Diseases) 数据集加载器,支持多标签分类(20 种视网膜病变)
# 数据集来源:MuReD Dataset (Porwal et al.)
#
# 与 ChestX-ray14 的差异:
#   1. 已预划分 train_data.csv / val_data.csv —— 直接使用,不再 80/20 自划分
#   2. 彩色眼底图 —— convert('RGB'),而非胸片的灰度 Grayscale(3)
#   3. 20 类长尾标签(含 NORMAL),标签向量 20 维
#   4. 图像为 png/tif 混合,文件名 = ID + 扩展名;部分 ID 自带 (NNNN) 前缀
#   5. 图像位于双层目录 images/images/

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

MURED_LABELS = [
    'DR', 'NORMAL', 'MH', 'ODC', 'TSLN', 'ARMD', 'DN', 'MYA', 'BRVO', 'ODP',
    'CRVO', 'CNV', 'RS', 'ODE', 'LS', 'CSR', 'HTR', 'ASR', 'CRS', 'OTHER'
]
# 20 类:糖尿病视网膜病变(DR)、正常(NORMAL)、黄斑裂孔(MH)、视盘可疑(ODC)、
# 中心凹反光缺失(TSLN)、年龄相关性黄斑变性(ARMD)、玻璃膜疣(DN)、近视黄斑病变(MYA)、
# 视网膜分支静脉阻塞(BRVO)、视盘异常(ODP)、视网膜中央静脉阻塞(CRVO)、
# 脉络膜新生血管(CNV)、视网膜疤痕(RS)、视盘水肿(ODE)、激光斑(LS)、
# 中心性浆液性视网膜病变(CSR)、其他高反射(HTR)、异常软反射(ASR)、
# 脉络膜皱褶(CRS)、其他(OTHER)。NORMAL 为显式标签(非"无标签")。


class MuReD(Dataset):
    """MuReD 多标签眼底数据集

    每张眼底图可能同时带多种病变标签;NORMAL 列表示正常(作为 20 类之一)。
    图像为 png/tif 混合,自动按 ID 查找扩展名,缩放到指定尺寸并转为 RGB 三通道。
    """

    def __init__(self, data_dir, split='train', transform=None, image_size=224):
        csv_name = 'train_data.csv' if split == 'train' else 'val_data.csv'
        csv_path = os.path.join(data_dir, csv_name)
        df = pd.read_csv(csv_path)

        # 双层 images 目录:data_dir/images/images/<ID>.{png,tif}
        self.image_dir = os.path.join(data_dir, 'images', 'images')
        self.transform = transform
        self.image_size = image_size

        self.filenames = []
        self.labels = []

        for _, row in df.iterrows():
            img_id = str(row['ID'])
            # 文件名 = ID + (.png 或 .tif);部分 ID 自带 (NNNN) 前缀,直接拼接即可
            img_path = None
            for ext in ('.png', '.tif'):
                cand = os.path.join(self.image_dir, img_id + ext)
                if os.path.exists(cand):
                    img_path = cand
                    break
            if img_path is None:
                continue

            self.filenames.append(img_path)

            label_vec = np.zeros(len(MURED_LABELS), dtype=np.float32)
            for i, name in enumerate(MURED_LABELS):
                val = row.get(name, 0)
                if pd.notna(val) and float(val) > 0:
                    label_vec[i] = 1.0
            self.labels.append(label_vec)

        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx]).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        if self.transform:
            img = self.transform(img)

        label = torch.from_numpy(self.labels[idx])
        return img, label

    def num_classes(self):
        return len(MURED_LABELS)

    @staticmethod
    def label_names():
        return MURED_LABELS
