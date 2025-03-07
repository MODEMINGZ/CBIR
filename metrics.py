import os
from config import config


class MetricCalculator:
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root  # 完整数据集路径
        self.total_relevant = 0

    def _get_class_from_path(self, img_path):
        """从图片路径中提取类别（最后一级目录名）"""
        # 示例路径：data/image/A0C573/xxx.jpg → 类别为 A0C573
        dir_path = os.path.dirname(img_path)
        return os.path.basename(dir_path)

    def calculate_all(self, query_path, results, search_time):
        # 提取查询图片的类别
        query_class = self._get_class_from_path(query_path)

        # 计算总相关图片数（遍历完整数据集中的同类目录）
        class_dir = os.path.join(self.dataset_root, query_class)
        if not os.path.exists(class_dir):
            total_relevant = 0
        else:
            total_relevant = len(
                [
                    f
                    for f in os.listdir(class_dir)
                    if f.lower().endswith(tuple(config.ALLOWED_EXTENSIONS))
                ]
            )

        # 统计相关结果数
        relevant = 0
        precision_at_k = []
        for k, path in enumerate(results, 1):
            result_class = self._get_class_from_path(path)
            if result_class == query_class:
                relevant += 1
                precision_at_k.append(relevant / k)

        # 计算指标
        recall = relevant / total_relevant if total_relevant > 0 else 0
        precision = relevant / len(results) if len(results) > 0 else 0
        ap = sum(precision_at_k) / total_relevant if total_relevant > 0 else 0
        self.total_relevant = total_relevant

        # 收集每个位置的精确率和召回率
        pr_data = []
        relevant = 0
        for k, path in enumerate(results, 1):
            result_class = self._get_class_from_path(path)
            if result_class == query_class:
                relevant += 1
            precision_k = relevant / k
            recall_k = relevant / total_relevant if total_relevant > 0 else 0
            pr_data.append((recall_k, precision_k))

        return {
            "pr_data": pr_data,
            "recall": float(round(recall, 4)),
            "precision": float(round(precision, 4)),
            "mAP": float(round(ap, 4)),
            "response_time": float(round(search_time, 1)),
            "total_relevant": total_relevant,
        }
