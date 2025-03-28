import numpy as np
from collections import defaultdict
import pickle
from scipy.sparse import csr_matrix
import os
from tqdm import tqdm


class TFIDFIndexer:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.document_freq = np.zeros(n_clusters)  # 文档频率
        self.total_documents = 0  # 总文档数
        self.idf = None  # 逆文档频率
        self.inverted_index = defaultdict(list)  # 倒排索引
        self.weighted_features = {}  # 加权后的特征向量

    def build_index(self, features_dict, kmeans):
        """
        构建倒排索引和计算TF-IDF权重
        :param features_dict: {image_path: feature_vector}
        :param kmeans: KMeans模型，用于获取视觉词汇的分配
        """
        print("构建倒排索引...")
        self.total_documents = len(features_dict)

        # 计算文档频率
        for img_path, feature_vector in tqdm(
            features_dict.items(), desc="计算文档频率"
        ):
            # 对于BoF向量，非零元素表示该视觉词汇在图像中出现
            word_present = feature_vector > 0
            self.document_freq += word_present

            # 构建倒排索引
            for word_id in np.where(word_present)[0]:
                self.inverted_index[word_id].append(img_path)

        # 计算IDF
        self.idf = np.log(self.total_documents / (self.document_freq + 1e-10))

        # 计算加权特征向量
        print("计算TF-IDF加权特征...")
        for img_path, feature_vector in tqdm(
            features_dict.items(), desc="应用TF-IDF权重"
        ):
            # TF已经在BoF向量中体现，直接与IDF相乘
            weighted_vector = feature_vector * self.idf
            # L2归一化
            norm = np.linalg.norm(weighted_vector)
            if norm > 0:
                weighted_vector /= norm
            self.weighted_features[img_path] = weighted_vector

    def save(self, save_dir):
        """保存索引和权重"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存IDF权重
        idf_path = os.path.join(save_dir, "idf_weights.pkl")
        with open(idf_path, "wb") as f:
            pickle.dump(
                {
                    "idf": self.idf,
                    "document_freq": self.document_freq,
                    "total_documents": self.total_documents,
                },
                f,
            )

        # 保存倒排索引
        index_path = os.path.join(save_dir, "inverted_index.pkl")
        with open(index_path, "wb") as f:
            pickle.dump(dict(self.inverted_index), f)

        # 保存加权特征
        features_path = os.path.join(save_dir, "weighted_features.pkl")
        with open(features_path, "wb") as f:
            pickle.dump(self.weighted_features, f)

    def load(self, save_dir):
        """加载索引和权重"""
        # 加载IDF权重
        idf_path = os.path.join(save_dir, "idf_weights.pkl")
        with open(idf_path, "rb") as f:
            weights_data = pickle.load(f)
            self.idf = weights_data["idf"]
            self.document_freq = weights_data["document_freq"]
            self.total_documents = weights_data["total_documents"]

        # 加载倒排索引
        index_path = os.path.join(save_dir, "inverted_index.pkl")
        with open(index_path, "rb") as f:
            self.inverted_index = defaultdict(list, pickle.load(f))

        # 加载加权特征
        features_path = os.path.join(save_dir, "weighted_features.pkl")
        with open(features_path, "rb") as f:
            self.weighted_features = pickle.load(f)

    def apply_weights(self, feature_vector):
        """对查询向量应用TF-IDF权重"""
        weighted_vector = feature_vector * self.idf
        # L2归一化
        norm = np.linalg.norm(weighted_vector)
        if norm > 0:
            weighted_vector /= norm
        return weighted_vector

    def search(self, query_vector, top_k=10):
        """
        使用倒排索引和TF-IDF权重进行搜索
        :param query_vector: 查询图像的特征向量
        :param top_k: 返回的最相似图像数量
        :return: [(image_path, similarity_score)]
        """
        # 应用TF-IDF权重到查询向量
        weighted_query = self.apply_weights(query_vector)

        # 使用倒排索引确定候选图像
        candidate_images = set()
        for word_id in np.where(query_vector > 0)[0]:
            candidate_images.update(self.inverted_index[word_id])

        if not candidate_images:
            return []

        # 计算相似度分数
        similarities = []
        for img_path in candidate_images:
            similarity = np.dot(weighted_query, self.weighted_features[img_path])
            similarities.append((img_path, similarity))

        # 排序并返回top-k结果
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
