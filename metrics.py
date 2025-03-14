import os
import numpy as np
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

    def calculate_ap(self, query_class, results, k=10):
        """计算AP (Average Precision)

        在前k个结果中计算AP：
        1. 对于每个正确结果（与查询图片同类），记录其位置的precision
        2. 将所有precision求和，除以该类别的总相关图片数

        Args:
            query_class: 查询图片的类别
            results: 检索结果路径列表
            k: 计算AP时考虑的结果数量（默认为10）

        Returns:
            float: AP值
        """
        relevant = 0
        precision_sum = 0

        # 只考虑前k个结果
        for i, path in enumerate(results[:k], 1):
            result_class = self._get_class_from_path(path)
            if result_class == query_class:
                relevant += 1
                # 计算当前位置的precision
                precision_at_i = relevant / i
                precision_sum += precision_at_i

        # AP = 正确结果位置的precision之和 / 总相关图片数
        ap = precision_sum / self.total_relevant if self.total_relevant > 0 else 0
        return ap

    def calculate_all(self, query_path, results, search_time):
        """计算单次查询的所有指标"""
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

        self.total_relevant = total_relevant

        # 计算AP@10作为mAP
        map_score = self.calculate_ap(query_class, results, k=10)

        # 计算最终的precision和recall（使用所有结果）
        relevant = sum(
            1 for path in results if self._get_class_from_path(path) == query_class
        )

        # recall = 返回的正确结果数 / 该类别的总图片数
        recall = relevant / total_relevant if total_relevant > 0 else 0
        # precision = 正确结果数 / 返回结果总数
        precision = relevant / len(results) if results else 0

        # 收集PR曲线数据
        pr_data = []
        relevant = 0
        for k, path in enumerate(results, 1):
            result_class = self._get_class_from_path(path)
            if result_class == query_class:
                relevant += 1
            precision_k = relevant / k
            recall_k = relevant / total_relevant if total_relevant > 0 else 0
            pr_data.append((recall_k, precision_k))

        # 计算11点插值PR曲线数据
        interpolated_pr_data = self.calculate_interpolated_pr_curve(pr_data)

        return {
            "pr_data": pr_data,
            "interpolated_pr_data": interpolated_pr_data,
            "recall": float(round(recall, 4)),
            "precision": float(round(precision, 4)),
            "mAP": float(round(map_score, 4)),
            "response_time": float(round(search_time, 1)),
            "total_relevant": total_relevant,
        }

    def calculate_interpolated_pr_curve(self, pr_data):
        """计算11点插值PR曲线

        在11个标准召回率点(0, 0.1, 0.2, ..., 1.0)上计算插值精确率

        Args:
            pr_data: 原始PR数据点列表 [(recall, precision),...]

        Returns:
            list: 11点插值后的PR数据 [(recall, precision),...]
        """
        if not pr_data:
            return []

        # 标准11点召回率
        standard_recalls = [i / 10 for i in range(11)]  # 0, 0.1, 0.2, ..., 1.0
        interpolated_precisions = []

        # 原始数据按召回率排序
        sorted_pr_data = sorted(pr_data, key=lambda x: x[0])
        recalls = [r for r, _ in sorted_pr_data]
        precisions = [p for _, p in sorted_pr_data]

        # 对每个标准召回率点，找到大于等于它的所有精确率的最大值
        for recall in standard_recalls:
            # 找到所有大于等于当前召回率的精确率
            precision_at_recall = [p for r, p in sorted_pr_data if r >= recall]

            # 如果没有找到，使用0
            if not precision_at_recall:
                interpolated_precisions.append(0)
            else:
                # 使用最大精确率
                interpolated_precisions.append(max(precision_at_recall))

        return list(zip(standard_recalls, interpolated_precisions))


class TestSetEvaluator:
    def __init__(self, dataset_root):
        self.calculator = MetricCalculator(dataset_root)
        self.dataset_root = dataset_root
        self.results = {
            "avg_precision": 0,
            "avg_recall": 0,
            "avg_map": 0,
            "avg_response_time": 0,
            "per_class_metrics": {},
            "total_queries": 0,
        }

    def evaluate_test_set(self, retrieval_func, test_images):
        """评估测试集上的性能

        Args:
            retrieval_func: 检索函数，接受图片路径返回(results, search_time)
            test_images: 测试图片路径列表
        """
        total_precision = 0
        total_recall = 0
        total_map = 0
        total_response_time = 0
        class_metrics = {}

        for query_path in test_images:
            # 执行检索
            results, search_time = retrieval_func(query_path)

            # 计算指标
            metrics = self.calculator.calculate_all(query_path, results, search_time)

            # 累加指标
            total_precision += metrics["precision"]
            total_recall += metrics["recall"]
            total_map += metrics["mAP"]
            total_response_time += metrics["response_time"]

            # 按类别收集指标
            query_class = self.calculator._get_class_from_path(query_path)
            if query_class not in class_metrics:
                class_metrics[query_class] = {
                    "precision": [],
                    "recall": [],
                    "map": [],
                    "response_time": [],
                }
            class_metrics[query_class]["precision"].append(metrics["precision"])
            class_metrics[query_class]["recall"].append(metrics["recall"])
            class_metrics[query_class]["map"].append(metrics["mAP"])
            class_metrics[query_class]["response_time"].append(metrics["response_time"])

        # 计算平均值
        num_queries = len(test_images)
        self.results = {
            "avg_precision": round(total_precision / num_queries, 4),
            "avg_recall": round(total_recall / num_queries, 4),
            "avg_map": round(total_map / num_queries, 4),
            "avg_response_time": round(total_response_time / num_queries, 1),
            "per_class_metrics": {
                cls: {
                    "avg_precision": round(np.mean(metrics["precision"]), 4),
                    "avg_recall": round(np.mean(metrics["recall"]), 4),
                    "avg_map": round(np.mean(metrics["map"]), 4),
                    "avg_response_time": round(np.mean(metrics["response_time"]), 1),
                    "num_queries": len(metrics["precision"]),
                }
                for cls, metrics in class_metrics.items()
            },
            "total_queries": num_queries,
        }

        return self.results
