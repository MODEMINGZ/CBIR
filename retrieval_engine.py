import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import pickle
from feature_extractor import FeatureExtractor
from config import config


class RetrievalEngine:
    def __init__(self, features_path, algo="SIFT", encoding="BoF", n_clusters=1000):
        """
        初始化检索引擎
        :param features_path: 特征文件路径或特征字典
        :param algo: 特征提取算法
        :param encoding: 特征编码方法
        :param n_clusters: 聚类数量
        """
        self.algo = algo
        self.encoding = encoding
        self.n_clusters = n_clusters

        # 加载词汇表和特征提取器
        self._load_vocabulary()

        # 加载特征数据
        self._load_features(features_path)
        self._build_index()

    def _load_features(self, features_path):
        """
        加载特征数据
        :param features_path: 特征文件路径或特征字典
        """
        if isinstance(features_path, str):
            # 从文件加载特征
            try:
                with open(features_path, "rb") as f:
                    features = pickle.load(f)
            except Exception as e:
                raise ValueError(f"加载特征文件失败: {str(e)}")
        else:
            # 直接使用传入的特征字典
            features = features_path

        # 初始化特征数据
        self.img_paths = list(features.keys())
        self.feature_matrix = np.array(list(features.values()))

    def _build_index(self):
        """构建KNN索引（首次检索时自动触发）"""
        self.nn = NearestNeighbors(n_neighbors=100, metric="cosine")
        self.nn.fit(self.feature_matrix)

    def _load_vocabulary(self):
        """加载词汇表和初始化特征提取器"""
        # 获取词汇表路径
        vocabulary_path = config.VOCABULARY_PATTERN.format(
            algo=self.algo, encoding=self.encoding, clusters=self.n_clusters
        )

        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor(
            algo=self.algo, encoding=self.encoding, n_clusters=self.n_clusters
        )

        # 加载词汇表
        try:
            with open(vocabulary_path, "rb") as f:
                vocabulary_data = pickle.load(f)
                if self.encoding in ["BoF", "VLAD"]:
                    self.feature_extractor.kmeans = vocabulary_data["kmeans"]
                elif self.encoding == "FV":
                    self.feature_extractor.gmm = vocabulary_data["kmeans"]
                self.feature_extractor.vocabulary = vocabulary_data["centers"]
        except Exception as e:
            raise ValueError(f"加载词汇表失败: {str(e)}")

    def encode_query_image(self, query_path):
        """
        对查询图像进行特征提取和编码
        :param query_path: 查询图像路径
        :return: 编码后的特征向量
        """
        # 提取特征
        raw_feature = self.feature_extractor.extract_single(query_path)
        if raw_feature is None:
            raise ValueError("特征提取失败")

        # 编码特征
        encoded_feature = self.feature_extractor.encode_feature(raw_feature)
        return encoded_feature

    def search(self, query_path, k=10):
        """
        执行检索
        :param query_path: 查询图片路径
        :param k: 返回结果数量
        :return: 元组 (结果路径列表, 相似度列表, 耗时ms)
        """
        start_time = time.time()

        # 提取并编码查询图像特征
        try:
            query_feature = self.encode_query_image(query_path)
            # 转换为二维数组
            query_feature = np.array(query_feature).reshape(1, -1)
        except Exception as e:
            raise ValueError(f"提取编码查询图像特征失败: {str(e)}")

        # 搜索最近邻
        distances, indices = self.nn.kneighbors(query_feature, n_neighbors=k)

        # 计算相似度（余弦相似度 = 1 - 余弦距离）
        # 注意：PyQt 信号要求传输的数据必须是 Python 原生类型（如 list）
        similarities = (1 - distances[0]).tolist()

        # 获取路径
        result_paths = [self.img_paths[i] for i in indices[0]]

        return result_paths, similarities, (time.time() - start_time) * 1000
