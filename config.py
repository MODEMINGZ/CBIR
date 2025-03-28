import os


class Config:
    # 数据集根目录
    DATASET_ROOT = os.path.join("data", "image")  # 完整数据集路径：data/image/
    TEST_ROOT = os.path.join("data", "test")  # 测试集路径：data/test/

    # 特征保存路径
    TEST_FEATURE_PATH = "test_features.pkl"
    FEATURE_PATTERN = os.path.join(
        "features", "features_{algo}_{encoding}_{clusters}.pkl"
    )
    VOCABULARY_PATTERN = os.path.join(
        "features", "vocabulary_{algo}_{encoding}_{clusters}.pkl"
    )
    TFIDF_INDEX_PATTERN = os.path.join(
        "features", "tfidf_index_{algo}_{encoding}_{clusters}"
    )

    # 特征编码配置
    DEFAULT_CLUSTERS = 1000  # 默认聚类数量
    ENCODING_METHODS = ["BoF", "VLAD", "FV"]  # 支持的编码方法

    # 允许的图片扩展名
    ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


# 单例模式全局配置
config = Config()
