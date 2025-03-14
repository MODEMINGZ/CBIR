import cv2
import os
import pickle
import numpy as np
from tqdm import tqdm
from config import config
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture


class FeatureExtractor:
    def __init__(self, algo="SIFT", encoding="BoF", n_clusters=1000):
        self.algo = algo
        self.encoding = encoding
        self.n_clusters = n_clusters

        # 算法兼容性处理
        if self.algo == "SIFT":
            self.extractor = cv2.SIFT_create()
        elif self.algo == "ORB":
            self.extractor = cv2.ORB_create(nfeatures=500)
        else:
            raise ValueError(f"不支持的算法: {algo}")

        self.features = {}
        self.kmeans = None
        self.vocabulary = None

    def _get_image_paths(self, root_dir):
        """递归获取目录下所有图片路径"""
        paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in config.ALLOWED_EXTENSIONS:
                    paths.append(os.path.join(dirpath, fname))
        return paths

    def extract_single(self, img_path):
        """提取单张图片特征"""
        # 标准化输入路径
        img_path = os.path.normpath(os.path.abspath(img_path)).lower()
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # 提取关键点和描述子
        kp, des = self.extractor.detectAndCompute(img, None)
        if des is None:
            return None

        return des

    def encode_feature(self, feature):
        """对特征进行编码"""
        if self.encoding == "BoF":
            return self.encode_bof(feature)
        elif self.encoding == "VLAD":
            return self.encode_vlad(feature)
        elif self.encoding == "FV":
            return self.encode_fv(feature)
        else:
            raise ValueError(f"不支持的编码方式: {self.encoding}")

    def encode_bof(self, feature):
        """BoF编码"""
        hist, _ = np.histogram(
            self.kmeans.predict(feature),
            bins=self.n_clusters,
            range=(0, self.n_clusters),
        )
        return normalize(hist.reshape(1, -1)).flatten()

    def encode_vlad(self, feature):
        """VLAD编码"""
        pred_labels = self.kmeans.predict(feature)
        vlad = np.zeros((self.n_clusters, feature.shape[1]))
        for i in range(self.n_clusters):
            if np.sum(pred_labels == i) > 0:
                vlad[i] = np.sum(
                    feature[pred_labels == i, :] - self.kmeans.cluster_centers_[i],
                    axis=0,
                )
        vlad = vlad.flatten()
        vlad = normalize(vlad.reshape(1, -1)).flatten()
        return vlad

    def extract_dataset(self, root_dir, save_path=None):
        """批量提取数据集特征"""
        paths = self._get_image_paths(root_dir)
        all_features = []
        for path in tqdm(paths, desc=f"提取特征 ({self.algo})"):
            feature = self.extract_single(path)
            if feature is not None:
                all_features.append(feature)

        # 构建词汇表
        self.build_vocabulary(np.vstack(all_features))

        # 编码特征
        for path, feature in zip(paths, all_features):
            encoded_feature = self.encode_feature(feature)
            self.features[path] = encoded_feature

        if save_path:
            self.save_features(save_path)
        return self.features

    def encode_fv(self, feature):
        """Fisher Vector编码"""
        # 计算GMM的后验概率
        posteriors = self.gmm.predict_proba(feature)

        # 初始化FV向量
        d = feature.shape[1]  # 特征维度
        K = self.n_clusters  # 聚类数量
        fv = np.zeros(K * d * 2)  # FV维度是聚类数量 * 特征维度 * 2

        # 获取GMM参数
        means = self.gmm.means_
        covs = self.gmm.covariances_
        weights = self.gmm.weights_

        # 计算一阶和二阶统计量
        for k in range(K):
            diff = feature - means[k]  # 一阶差异

            # 归一化
            norm_diff = diff / np.sqrt(covs[k])

            # 加权
            weighted_diff = posteriors[:, k].reshape(-1, 1) * norm_diff

            # 一阶统计量
            fv_1 = np.sum(weighted_diff, axis=0) / np.sqrt(weights[k])

            # 二阶统计量
            fv_2 = np.sum(
                posteriors[:, k].reshape(-1, 1) * (norm_diff**2 - 1), axis=0
            ) / np.sqrt(2 * weights[k])

            # 填充FV向量
            fv[k * d : (k + 1) * d] = fv_1
            fv[K * d + k * d : K * d + (k + 1) * d] = fv_2

        # 幂归一化和L2归一化
        fv = np.sign(fv) * np.sqrt(np.abs(fv))  # 幂归一化
        fv = normalize(fv.reshape(1, -1)).flatten()  # L2归一化

        return fv

    def build_vocabulary(self, features):
        """构建词汇表"""
        print("构建词汇表...")

        if self.encoding in ["BoF", "VLAD"]:
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.kmeans.fit(features)
            self.vocabulary = self.kmeans.cluster_centers_
        elif self.encoding == "FV":
            # 对于FV，我们使用GMM而不是K-means
            print("训练高斯混合模型...")
            # 如果特征数量太多，随机抽样以加速训练
            if features.shape[0] > 100000:
                idx = np.random.choice(features.shape[0], 100000, replace=False)
                sample_features = features[idx]
            else:
                sample_features = features

            self.gmm = GaussianMixture(
                n_components=self.n_clusters,
                covariance_type="diag",  # 使用对角协方差矩阵以减少计算量
                random_state=42,
                max_iter=100,
                verbose=1,
            )
            self.gmm.fit(sample_features)
            self.vocabulary = self.gmm.means_

    def save_features(self, save_path):
        """保存特征到文件"""
        with open(save_path, "wb") as f:
            pickle.dump(self.features, f)

    def load_features(self, load_path):
        """从文件加载特征"""
        with open(load_path, "rb") as f:
            self.features = pickle.load(f)
