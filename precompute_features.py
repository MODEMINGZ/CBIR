from feature_extractor import FeatureExtractor
from config import config
import os


def main():
    # 创建特征保存目录
    os.makedirs("features", exist_ok=True)

    # 支持的所有算法列表
    algorithms = ["SIFT", "ORB"]

    for algo in algorithms:
        print(f"\n正在提取{algo}特征...")
        try:
            # 初始化提取器
            extractor = FeatureExtractor(algo=algo)

            # 生成特征保存路径（动态替换算法名）
            save_path = config.FEATURE_PATTERN.format(algo=algo)

            # 提取并保存特征
            extractor.extract_dataset(config.DATASET_ROOT, save_path=save_path)
            print(f"{algo}特征已保存至: {save_path}")
        except Exception as e:
            print(f"{algo}特征提取失败: {str(e)}")


if __name__ == "__main__":
    main()
