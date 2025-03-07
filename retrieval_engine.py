import numpy as np
from sklearn.neighbors import NearestNeighbors
import time


class RetrievalEngine:
    def __init__(self, features):
        """
        :param features: 字典格式 { "图片路径": 特征向量 }
        """
        self.img_paths = list(features.keys())
        self.feature_matrix = np.array(list(features.values()))
        self._build_index()

    def _build_index(self):
        """构建KNN索引（首次检索时自动触发）"""
        self.nn = NearestNeighbors(n_neighbors=100, metric="cosine")
        self.nn.fit(self.feature_matrix)

    def search(self, query_feature, k=10):
        """
        执行检索
        :param query_feature: 查询图片的特征向量
        :param k: 返回结果数量
        :return: 元组 (结果路径列表, 相似度列表, 耗时ms)
        """
        start_time = time.time()

        # 转换为二维数组
        query_feature = np.array(query_feature).reshape(1, -1)

        # 搜索最近邻
        distances, indices = self.nn.kneighbors(query_feature, n_neighbors=k)

        # 计算相似度（余弦相似度 = 1 - 余弦距离）
        # 注意：PyQt 信号要求传输的数据必须是 Python 原生类型（如 list）
        similarities = (1 - distances[0]).tolist()

        # 获取路径
        result_paths = [self.img_paths[i] for i in indices[0]]

        return result_paths, similarities, (time.time() - start_time) * 1000
