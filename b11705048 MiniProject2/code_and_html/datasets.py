# datasets.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import os

# 定義多標籤數據集，繼承自 torch.utils.data.Dataset
class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, img_dir, class_mapping, transform=None):
        """
        初始化數據集，將 dataframe、圖片路徑、標籤映射和數據增強變換等作為參數。
        
        :param dataframe: 包含圖像文件名和標籤的 pandas DataFrame
        :param img_dir: 圖像文件所在的目錄
        :param class_mapping: 類別映射字典，用於將類別名稱轉換為索引
        :param transform: 圖像增強的變換操作（可選）
        """
        self.dataframe = dataframe.reset_index(drop=True)  # 重設索引，確保數據的順序一致
        self.img_dir = img_dir  # 圖像目錄
        self.transform = transform  # 圖像增強操作
        self.class_mapping = class_mapping  # 類別映射
        self.num_classes = len(class_mapping)  # 類別數量

    def __len__(self):
        """
        返回數據集的長度，即樣本數量
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        根據索引加載圖片及其對應的標籤。
        
        :param idx: 索引
        :return: 加載的圖片及其標籤
        """
        img_path = f"{self.img_dir}/{self.dataframe.loc[idx, 'filename']}"  # 圖片路徑
        image = Image.open(img_path).convert("RGB")  # 讀取圖片並轉為 RGB 模式

        if self.transform:
            image = self.transform(image)  # 如果有定義變換操作，對圖像進行處理

        # 提取多標籤，這裡假設標籤列名為 class_0, class_1, ..., class_n
        labels = self.dataframe.loc[idx, [f'class_{i}' for i in range(self.num_classes)]].values.astype('float32')
        labels = torch.tensor(labels)  # 將標籤轉為 tensor 類型

        return image, labels  # 返回圖像和標籤
    
# 定義測試數據集，繼承自 torch.utils.data.Dataset
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None, extensions=['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
        """
        初始化測試數據集，包含圖像路徑、增強操作及文件擴展名過濾器。
        
        :param img_dir: 圖像目錄
        :param transform: 圖像增強的變換操作（可選）
        :param extensions: 允許的圖像擴展名列表
        """
        self.img_dir = img_dir  # 圖像目錄
        self.transform = transform  # 圖像增強操作
        self.extensions = extensions  # 允許的擴展名
        self.filenames = self._get_filenames()  # 取得所有符合條件的文件名

    def _get_filenames(self):
        """
        獲取指定目錄下的所有圖像文件名，並過濾出指定擴展名的文件。
        """
        try:
            files = os.listdir(self.img_dir)  # 獲取目錄下所有文件
        except FileNotFoundError:
            print(f"錯誤: 測試資料夾 '{self.img_dir}' 不存在。請確認路徑是否正確。")
            return []
        img_files = [f for f in files if os.path.splitext(f)[1].lower() in self.extensions]  # 篩選出圖像文件
        img_files.sort()  # 可選：按名稱排序
        return img_files

    def __len__(self):
        """
        返回測試數據集的長度，即測試圖像的數量
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        根據索引加載測試圖像並返回。
        
        :param idx: 索引
        :return: 圖像的文件名和圖像本身
        """
        filename = self.filenames[idx]  # 獲取文件名
        img_path = os.path.join(self.img_dir, filename)  # 圖像完整路徑
        try:
            image = Image.open(img_path).convert("RGB")  # 讀取圖像並轉為 RGB 模式
        except Exception as e:
            print(f"錯誤: 無法打開圖像 '{img_path}'. 錯誤信息: {e}")
            # 返回一個全黑的圖像或其他處理方式
            image = Image.new('RGB', (224, 224), (0, 0, 0))  # 返回一個黑色的圖像，避免程式崩潰

        if self.transform:
            image = self.transform(image)  # 如果有定義變換操作，對圖像進行處理

        return filename, image  # 返回圖像的文件名和圖像
