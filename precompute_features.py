from feature_extractor import FeatureExtractor
from config import config
import os
import pickle


def main(algo=None, encoding=None, n_clusters=None):
    """
    生成特征文件
    :param algo: 指定的特征提取算法，如果为None则生成所有算法的特征
    :param encoding: 指定的编码方法，如果为None则生成所有编码方法的特征
    :param n_clusters: 指定的聚类数量，如果为None则使用默认值
    """
    # 创建特征保存目录
    os.makedirs("features", exist_ok=True)

    # 确定要处理的参数范围
    algorithms = [algo] if algo else ["SIFT", "ORB"]
    encodings = [encoding] if encoding else config.ENCODING_METHODS
    cluster_sizes = [n_clusters] if n_clusters else [config.DEFAULT_CLUSTERS]

    for current_algo in algorithms:
        for current_encoding in encodings:
            for current_clusters in cluster_sizes:
                print(
                    f"\n正在提取{current_algo}特征 (编码: {current_encoding}, 聚类数: {current_clusters})..."
                )
                try:
                    # 初始化提取器
                    extractor = FeatureExtractor(
                        algo=current_algo,
                        encoding=current_encoding,
                        n_clusters=current_clusters,
                    )

                    # 生成特征和词汇表保存路径
                    save_path = config.FEATURE_PATTERN.format(
                        algo=algo, encoding=encoding, clusters=n_clusters
                    )
                    vocab_path = config.VOCABULARY_PATTERN.format(
                        algo=algo, encoding=encoding, clusters=n_clusters
                    )

                    # 提取并保存特征
                    extractor.extract_dataset(config.DATASET_ROOT, save_path=save_path)

                    # 保存词汇表
                    with open(vocab_path, "wb") as f:
                        pickle.dump(
                            {
                                "kmeans": extractor.kmeans,
                                "centers": extractor.vocabulary,
                            },
                            f,
                        )

                    print(f"{algo}+{encoding}特征已保存至: {save_path}")
                    print(f"词汇表已保存至: {vocab_path}")
                except Exception as e:
                    print(f"{algo}特征提取失败: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="特征提取和编码工具")
    parser.add_argument("--algo", choices=["SIFT", "ORB"], help="指定特征提取算法")
    parser.add_argument(
        "--encoding", choices=config.ENCODING_METHODS, help="指定编码方法"
    )
    parser.add_argument("--clusters", type=int, help="指定聚类数量")
    args = parser.parse_args()

    main(args.algo, args.encoding, args.clusters)
