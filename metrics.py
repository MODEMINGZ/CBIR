from collections import defaultdict
import numpy as np
import os
from config import config


class MetricCalculator:
    def __init__(self, test_root):
        self.class_info = self._build_class_info(test_root)
        self.total_relevant = 0  # 新增：保存总相关图片数

    def _build_class_info(self, root_dir):
        class_dict = {}
        for dirpath, _, filenames in os.walk(root_dir):
            # 获取相对于测试集根目录的路径作为类别
            relative_path = os.path.relpath(dirpath, root_dir)
            class_name = relative_path.replace(os.sep, "_")

            for fname in filenames:
                if fname.lower().endswith(tuple(config.ALLOWED_EXTENSIONS)):
                    img_path = os.path.join(dirpath, fname)
                    class_dict[img_path] = class_name
        return class_dict

    def calculate_all(self, query_path, results, search_time):
        # 获取查询图片的类别
        query_class = self.class_info.get(query_path, "unknown")

        # 计算总相关图片数（在完整数据集中）
        self.total_relevant = sum(
            1 for path in self.class_info.values() if path == query_class
        )

        # 计算召回率、精度等
        relevant = sum(
            1 for path in results if self.class_info.get(path) == query_class
        )
        recall = relevant / self.total_relevant if self.total_relevant > 0 else 0
        precision = relevant / len(results) if len(results) > 0 else 0
        ap = self._average_precision(query_class, results)

        return {
            "recall": float(round(recall, 4)),
            "precision": float(round(precision, 4)),
            "mAP": float(round(ap, 4)),
            "response_time": float(round(search_time, 1)),
            "total_relevant": self.total_relevant,  # 新增返回总相关数
        }

    def _basic_metrics(self, query_class, results, total_relevant):
        relevant = sum(
            1 for path in results if self.class_info.get(path) == query_class
        )
        recall = relevant / total_relevant if total_relevant > 0 else 0
        precision = relevant / len(results) if len(results) > 0 else 0
        return recall, precision

    def _average_precision(self, query_class, results):
        """计算单次查询的AP（Average Precision）"""
        relevant_count = 0
        precision_at_k = []

        for k, path in enumerate(results, 1):
            if self.class_info.get(path) == query_class:
                relevant_count += 1
                precision_at_k.append(relevant_count / k)

        if relevant_count == 0:
            return 0.0

        return sum(precision_at_k) / relevant_count
