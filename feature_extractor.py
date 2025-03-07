import cv2
import os
import pickle
import numpy as np
from tqdm import tqdm
from config import config


class FeatureExtractor:
    def __init__(self, algo="SIFT"):
        self.algo = algo

        # 算法兼容性处理
        if self.algo == "SIFT":
            self.extractor = cv2.SIFT_create()
        elif self.algo == "ORB":  # 使用专利过期的ORB替代SURF
            self.extractor = cv2.ORB_create(nfeatures=500)
        else:
            raise ValueError(f"不支持的算法: {algo}")

        self.features = {}

    def _get_image_paths(self, root_dir):
        """递归获取目录下所有图片路径"""
        paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in config.ALLOWED_EXTENSIONS:
                    paths.append(os.path.join(dirpath, fname))
        return paths

    def extract_single(self, img_path):
        # 标准化输入路径
        img_path = os.path.normpath(os.path.abspath(img_path)).lower()
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        """提取单张图片特征"""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # 提取关键点和描述子
        kp, des = self.extractor.detectAndCompute(img, None)
        if des is None:
            return None

        # 取特征向量的均值
        return np.mean(des, axis=0)

    def extract_dataset(self, root_dir, save_path=None):
        """批量提取数据集特征"""
        paths = self._get_image_paths(root_dir)
        for path in tqdm(paths, desc=f"提取特征 ({self.algo})"):
            feature = self.extract_single(path)
            if feature is not None:
                self.features[path] = feature

        if save_path:
            self.save_features(save_path)
        return self.features

    def save_features(self, save_path):
        """保存特征到文件"""
        with open(save_path, "wb") as f:
            pickle.dump(self.features, f)

    def load_features(self, load_path):
        """从文件加载特征"""
        with open(load_path, "rb") as f:
            self.features = pickle.load(f)
