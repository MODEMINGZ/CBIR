# main.py
import sys
import os
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap
from feature_extractor import FeatureExtractor
from retrieval_engine import RetrievalEngine
from metrics import MetricCalculator
from config import config


# ======================== 多线程检索类 ========================
class SearchThread(QThread):
    search_finished = pyqtSignal(
        list, list, float, dict
    )  # 信号：结果路径、相似度、耗时、指标

    def __init__(self, engine, metric_calculator, query_path, query_feature, k):
        super().__init__()
        self.engine = engine
        self.metric_calculator = metric_calculator
        self.query_path = query_path
        self.query_feature = query_feature
        self.k = k

    def run(self):
        """执行检索并计算指标"""
        try:
            # 执行检索
            result_paths, similarities, search_time = self.engine.search(
                self.query_feature, self.k
            )

            # 计算性能指标
            metrics = self.metric_calculator.calculate_all(
                self.query_path, result_paths, search_time
            )

            # 发送结果信号
            self.search_finished.emit(result_paths, similarities, search_time, metrics)
        except Exception as e:
            print(f"检索出错: {str(e)}")


# ======================== 主界面类 ========================
class ImageRetrievalUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像检索系统_2212190121")
        self.current_image_path = None  # 当前上传的图片路径

        # ---------- 初始化UI布局 ----------
        self.init_ui()

        # ---------- 初始化后端模块 ----------
        self.init_backend()

        # ---------- 连接信号与槽 ----------
        self.search_btn.clicked.connect(self.on_search_clicked)
        self.upload_btn.clicked.connect(self.upload_image)
        self.feature_algo.currentIndexChanged.connect(self.on_algorithm_changed)
        self.copy_btn.clicked.connect(self.copy_metrics)

    # ----------------- 初始化方法 -----------------
    def init_ui(self):
        # 获取屏幕尺寸
        screen_geometry = QApplication.desktop().screenGeometry()
        screen_width, screen_height = screen_geometry.width(), screen_geometry.height()

        # 计算窗口初始位置，使其居中
        window_width, window_height = 1200, 700
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.setGeometry(x, y, window_width, window_height)

        # 主布局容器
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # 左侧控制面板
        left_panel = QWidget()
        left_panel.setFixedWidth(280)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)

        # 图片预览区域
        self.image_preview = QLabel()
        self.image_preview.setFixedSize(260, 260)
        self.image_preview.setStyleSheet("border: 2px dashed #aaa;")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setText("待检索图片预览")

        # 上传按钮
        self.upload_btn = QPushButton("上传图片")
        self.upload_btn.setFixedHeight(40)

        # 参数设置区域
        settings_group = QGroupBox("检索设置")
        settings_layout = QFormLayout()
        settings_layout.setVerticalSpacing(15)

        self.result_num = QSpinBox()
        self.result_num.setRange(1, 100)
        self.result_num.setValue(10)

        self.feature_algo = QComboBox()
        self.feature_algo.addItems(["SIFT", "ORB"])

        settings_layout.addRow("返回数量:", self.result_num)
        settings_layout.addRow("特征算法:", self.feature_algo)
        settings_group.setLayout(settings_layout)

        # 检索按钮
        self.search_btn = QPushButton("开始检索")
        self.search_btn.setFixedHeight(40)
        self.search_btn.setStyleSheet("background-color: #4CAF50; color: white;")

        left_layout.addWidget(self.image_preview)
        left_layout.addWidget(self.upload_btn)
        left_layout.addWidget(settings_group)
        left_layout.addWidget(self.search_btn)

        # 中间结果展示
        mid_panel = QScrollArea()
        mid_panel.setWidgetResizable(True)
        mid_content = QWidget()
        self.mid_layout = QGridLayout(mid_content)
        self.mid_layout.setAlignment(Qt.AlignTop)
        mid_panel.setWidget(mid_content)

        # 右侧性能面板
        right_panel = QScrollArea()
        right_panel.setWidgetResizable(True)
        right_content = QWidget()
        right_layout = QFormLayout(right_content)
        right_layout.setVerticalSpacing(15)
        self.current_algorithm = QLabel("SIFT")

        # 性能指标
        self.current_query = QLabel("---")
        self.recall = QLabel("---")
        self.precision = QLabel("---")
        self.response_time = QLabel("---")
        self.map = QLabel("---")
        self.total_relevant = QLabel("---")

        metrics = [
            ("召回率 (Recall)", self.recall),
            ("精确率 (Precision)", self.precision),
            ("mAP", self.map),
            ("响应时间 (ms)", self.response_time),
        ]

        right_layout.addRow("当前算法:", self.current_algorithm)
        right_layout.addRow("查询图片:", self.current_query)
        right_layout.addRow("总相关图片:", self.total_relevant)
        for metric in metrics:
            right_layout.addRow(metric[0], metric[1])

        self.copy_btn = QPushButton("复制性能数据")
        right_layout.addRow(self.copy_btn)

        right_panel.setWidget(right_content)

        # 添加主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(mid_panel, stretch=3)
        main_layout.addWidget(right_panel, stretch=1)

        # 设置样式
        self.setStyleSheet(
            """
            QWidget {
                font-family: Segoe UI;
                font-size: 14px;
            }
            QGroupBox {
                border: 1px solid #ddd;
                margin-top: 10px;
                padding-top: 15px;
            }
            QScrollArea {
                border: none;
            }
            QLabel[objectName^="metric_"] {
                color: #2196F3;
                font-weight: bold;
            }
        """
        )

    def init_backend(self):
        """初始化特征提取器、检索引擎和指标计算器"""
        try:
            # 加载特征数据
            self.feature_extractor = FeatureExtractor()
            self.feature_extractor.load_features(
                config.FEATURE_PATTERN.format(algo="SIFT")
            )

            # 初始化检索引擎
            self.retrieval_engine = RetrievalEngine(self.feature_extractor.features)

            # 初始化性能计算器
            self.metric_calculator = MetricCalculator(config.TEST_ROOT)
        except FileNotFoundError:
            QMessageBox.critical(
                self,
                "错误",
                "未找到特征文件，请先运行precompute_features.py生成特征数据！",
            )
            sys.exit(1)

    # ----------------- 核心逻辑 -----------------
    def copy_metrics(self):
        """复制完整的性能报告"""
        text = f"=== 检索性能报告 ===\n"
        text += f"算法类型: {self.feature_algo.currentText()}\n"
        text += f"查询图片: {os.path.basename(self.current_image_path) if self.current_image_path else '未知'}\n"
        text += f"总相关图片: {self.metric_calculator.total_relevant}\n"
        text += f"召回率: {self.recall.text()}（相关结果/{self.metric_calculator.total_relevant})\n"
        text += (
            f"精确率: {self.precision.text()}（相关结果/{self.result_num.value()})\n"
        )
        text += f"mAP: {self.map.text()}\n"
        text += f"响应时间: {self.response_time.text()}"

        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        QMessageBox.information(self, "已复制", "性能数据已复制到剪贴板")

    def on_algorithm_changed(self):
        """算法切换事件处理"""
        algo = self.feature_algo.currentText()

        try:
            # 动态生成特征文件路径
            feature_path = config.FEATURE_PATTERN.format(algo=algo)

            # 加载特征
            self.feature_extractor = FeatureExtractor(algo=algo)
            self.feature_extractor.load_features(feature_path)

            # 更新检索引擎
            self.retrieval_engine = RetrievalEngine(self.feature_extractor.features)

            # 更新界面显示
            self.statusBar().showMessage(f"已切换到{algo}算法", 3000)

        except FileNotFoundError:
            QMessageBox.critical(
                self,
                "错误",
                f"未找到{algo}算法的特征文件！\n请先运行precompute_features.py生成",
            )
            # 回退到默认算法
            self.feature_algo.blockSignals(True)  # 防止递归触发
            self.feature_algo.setCurrentText("SIFT")
            self.feature_algo.blockSignals(False)
            self.on_algorithm_changed()  # 强制刷新

    def upload_image(self):
        """上传图片并显示预览"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", config.TEST_ROOT, "Images (*.png *.jpg *.bmp)"
        )
        if not file_path:
            return

        # 显示预览
        self.current_image_path = file_path
        pixmap = QPixmap(file_path)
        self.image_preview.setPixmap(
            pixmap.scaled(
                self.image_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def on_search_clicked(self):
        """点击检索按钮的槽函数"""
        if not self.current_image_path:
            QMessageBox.warning(self, "提示", "请先上传查询图片！")
            return

        # 提取特征
        try:
            query_feature = self.feature_extractor.extract_single(
                self.current_image_path
            )
            if query_feature is None:
                raise ValueError("无法提取该图片特征")
        except Exception as e:
            QMessageBox.warning(self, "错误", str(e))
            return

        # 禁用按钮防止重复点击
        self.search_btn.setEnabled(False)
        self.search_btn.setText("检索中...")

        # 更新算法显示
        self.current_algorithm.setText(self.feature_algo.currentText())

        # 创建并启动检索线程
        self.thread = SearchThread(
            self.retrieval_engine,
            self.metric_calculator,
            self.current_image_path,
            query_feature,
            self.result_num.value(),
        )
        self.thread.search_finished.connect(self.on_search_finished)
        self.thread.start()

    def on_search_finished(self, result_paths, similarities, search_time, metrics):
        """接收检索结果并更新界面"""
        # 恢复按钮状态
        self.search_btn.setEnabled(True)
        self.search_btn.setText("开始检索")

        # 显示检索结果
        self.display_results(result_paths, similarities)

        # 更新性能指标
        self.update_metrics(metrics, search_time)
        # 显示基础统计信息
        self.current_query.setText(os.path.basename(self.current_image_path))
        self.total_relevant.setText(
            f"相关图片总数: {self.metric_calculator.total_relevant}"
        )

    # ----------------- 界面更新方法 -----------------
    def display_results(self, result_paths, similarities):
        """在中间区域显示检索结果"""
        # 清空旧内容
        for i in reversed(range(self.mid_layout.count())):
            self.mid_layout.itemAt(i).widget().deleteLater()

        # 动态添加结果
        row, col = 0, 0
        max_col = 2  # 每行2列
        for path, sim in zip(result_paths, similarities):
            # 创建缩略图
            thumbnail = QLabel()
            pixmap = QPixmap(path).scaled(200, 200, Qt.KeepAspectRatio)
            thumbnail.setPixmap(pixmap)

            # 创建信息标签
            info = QLabel(f"{os.path.basename(path)}\n相似度: {sim*100:.3f}%")
            info.setAlignment(Qt.AlignCenter)

            # 添加到布局
            container = QVBoxLayout()
            container.addWidget(thumbnail)
            container.addWidget(info)

            widget = QWidget()
            widget.setLayout(container)
            self.mid_layout.addWidget(widget, row, col)

            # 更新行列索引
            col += 1
            if col >= max_col:
                col = 0
                row += 1

    def update_metrics(self, metrics, search_time):
        """更新右侧性能指标"""
        self.recall.setText(
            f"{metrics['recall']*100:.1f}% ({metrics['total_relevant']}相关)"
        )
        self.precision.setText(f"{metrics['precision']*100:.1f}%")
        self.map.setText(f"{metrics['mAP']:.3f}")
        self.response_time.setText(f"{metrics['response_time']} ms")


# ======================== 启动程序 ========================
if __name__ == "__main__":
    # 高DPI适配
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)

    # 检查特征文件是否存在
    from config import config
    from feature_extractor import FeatureExtractor

    # 获取所有支持的算法
    supported_algos = ["SIFT", "ORB"]  # 与precompute_features.py中的algorithms一致

    # 检查所有算法的特征文件是否存在
    missing_algos = []
    for algo in supported_algos:
        feature_path = config.FEATURE_PATTERN.format(algo=algo)
        if not os.path.exists(feature_path):
            missing_algos.append(algo)

    # 如果有缺失的特征文件
    if missing_algos:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("特征文件缺失")
        msg.setText(
            f"以下算法的特征文件未找到：{', '.join(missing_algos)}\n"
            "是否需要现在生成？\n"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        reply = msg.exec_()

        if reply == QMessageBox.Yes:
            # 执行特征预生成
            from subprocess import run

            try:
                # 调用预计算脚本（显示控制台窗口）
                run(["python", "precompute_features.py"], check=True)

                # 再次验证是否生成成功
                for algo in missing_algos:
                    path = config.FEATURE_PATTERN.format(algo=algo)
                    if not os.path.exists(path):
                        QMessageBox.critical(
                            None, "错误", f"生成{algo}特征失败！请检查控制台输出"
                        )
                        sys.exit(1)
            except Exception as e:
                QMessageBox.critical(
                    None,
                    "运行错误",
                    f"特征生成失败：{str(e)}\n请手动运行precompute_features.py",
                )
                sys.exit(1)
        else:
            sys.exit(0)

    # 启动主窗口
    window = ImageRetrievalUI()
    window.show()
    sys.exit(app.exec_())
