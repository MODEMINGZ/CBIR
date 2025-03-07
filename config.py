import os


class Config:
    # 数据集根目录
    DATASET_ROOT = os.path.join("data", "image")  # 完整数据集路径：data/image/
    TEST_ROOT = os.path.join("data", "test")  # 测试集路径：data/test/

    # 特征保存路径
    TEST_FEATURE_PATH = "test_features.pkl"  # 测试集特征（可选）
    FEATURE_PATTERN = os.path.join("features", "features_{algo}.pkl")
    # 允许的图片扩展名
    ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


# 单例模式全局配置
config = Config()
