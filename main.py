# main.py
import sys
import os
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QEvent
from PyQt5.QtGui import QPixmap
from feature_extractor import FeatureExtractor
from retrieval_engine import RetrievalEngine
from metrics import MetricCalculator
from config import config
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PRCanvas(FigureCanvas):
    """PR曲线绘图组件"""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=80)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Recall")
        self.ax.set_ylabel("Precision")
        self.ax.set_title("PR Curve")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, linestyle="--", alpha=0.7)

    def update_plot(self, pr_data, interpolated_pr_data=None):
        """根据数据更新曲线

        Args:
            pr_data: 原始PR数据点列表 [(recall, precision),...]
            interpolated_pr_data: 11点插值后的PR数据 [(recall, precision),...]
        """
        self.ax.clear()
        if len(pr_data) == 0:
            return

        # 绘制原始PR曲线
        recalls, precisions = zip(*pr_data)
        self.ax.plot(
            recalls,
            precisions,
            "b-",
            marker="o",
            markersize=3,
            label="Original PR Curve",
            alpha=0.6,
        )

        # 绘制插值PR曲线
        if interpolated_pr_data and len(interpolated_pr_data) > 0:
            int_recalls, int_precisions = zip(*interpolated_pr_data)
            self.ax.plot(
                int_recalls,
                int_precisions,
                "r--",
                marker="s",
                markersize=4,
                label="11-point Interpolation",
            )

        self.ax.set_xlabel("Recall")
        self.ax.set_ylabel("Precision")
        self.ax.set_title("PR Curve")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, linestyle="--", alpha=0.7)
        self.ax.legend()
        self.fig.tight_layout()
        self.draw()


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
        self.analyze_btn.clicked.connect(self.analyze_test_set)

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

        # 统计测试集按钮
        self.analyze_btn = QPushButton("统计测试集")
        self.analyze_btn.setFixedHeight(30)
        self.analyze_btn.setStyleSheet("background-color: #2196F3; color: white;")

        left_layout.addWidget(self.image_preview)
        left_layout.addWidget(self.upload_btn)
        left_layout.addWidget(settings_group)
        left_layout.addWidget(self.search_btn)
        left_layout.addWidget(self.analyze_btn)

        # 中间结果展示
        mid_panel = QScrollArea()
        mid_panel.setWidgetResizable(True)
        mid_content = QWidget()
        self.mid_layout = QGridLayout(mid_content)
        self.mid_layout.setAlignment(Qt.AlignTop)
        mid_panel.setWidget(mid_content)

        # ================= 右侧面板 =================
        self.right_panel = QScrollArea()
        self.right_panel.setWidgetResizable(True)
        self.right_panel.setMinimumWidth(320)  # 固定最小宽度

        # 主容器
        right_content = QWidget()
        right_content_layout = QVBoxLayout(right_content)
        right_content_layout.setContentsMargins(5, 5, 5, 5)

        # ----------------- 性能指标分组 -----------------
        self.performance_group = QGroupBox("性能指标")
        perf_layout = QFormLayout()
        perf_layout.setVerticalSpacing(10)

        # 算法和查询信息
        self.current_algorithm = QLabel("SIFT")
        self.current_query = QLabel("---")
        self.current_query.setWordWrap(True)
        self.total_relevant = QLabel("---")

        # 测试集统计
        self.test_total_images = QLabel("---")
        self.test_total_classes = QLabel("---")
        self.test_avg_per_class = QLabel("---")

        # 核心指标
        self.recall = QLabel("---")
        self.precision = QLabel("---")
        self.map = QLabel("---")
        self.response_time = QLabel("---")

        # 添加到布局
        perf_layout.addRow("当前算法:", self.current_algorithm)
        perf_layout.addRow("查询图片:", self.current_query)
        perf_layout.addRow("总相关数:", self.total_relevant)
        perf_layout.addRow(QLabel(""))  # 空行分隔

        # 测试集统计区
        perf_layout.addRow("测试集总图片:", self.test_total_images)
        perf_layout.addRow("测试集类别数:", self.test_total_classes)
        perf_layout.addRow("平均每类图片:", self.test_avg_per_class)
        perf_layout.addRow(QLabel(""))  # 空行分隔

        # 检索性能区
        perf_layout.addRow("召回率:", self.recall)
        perf_layout.addRow("精确率:", self.precision)
        perf_layout.addRow("mAP:", self.map)
        perf_layout.addRow("响应时间:", self.response_time)
        self.performance_group.setLayout(perf_layout)

        # ----------------- PR曲线区域 -----------------
        self.pr_canvas = PRCanvas()  # 使用之前定义的PR曲线组件

        # ----------------- 操作按钮 -----------------
        self.copy_btn = QPushButton("复制性能数据")
        self.copy_btn.setFixedHeight(30)

        # 整合所有组件
        right_content_layout.addWidget(self.performance_group)
        right_content_layout.addWidget(self.pr_canvas)
        right_content_layout.addWidget(self.copy_btn)
        right_content_layout.addStretch()  # 底部留空

        # 设置滚动区域内容
        self.right_panel.setWidget(right_content)

        # 添加主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(mid_panel, stretch=3)
        main_layout.addWidget(self.right_panel, stretch=1)

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
        # 初始化性能计算器
        self.metric_calculator = MetricCalculator(config.DATASET_ROOT)

        # 初始化特征提取器（延迟到实际使用时）
        self.feature_extractor = None
        self.retrieval_engine = None

    def ensure_algorithm_support(self, algo):
        """确保算法支持并加载特征"""
        feature_path = config.FEATURE_PATTERN.format(algo=algo)

        if not os.path.exists(feature_path):
            reply = QMessageBox.question(
                self,
                "特征文件缺失",
                f"未找到{algo}算法的特征文件，是否现在生成？",
                QMessageBox.Yes | QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                from subprocess import run

                try:
                    run(["python", "precompute_features.py"], check=True)
                    if not os.path.exists(feature_path):
                        raise FileNotFoundError(f"生成{algo}特征失败")
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"特征生成失败: {str(e)}")
                    return False
            else:
                return False

        try:
            # 加载特征
            self.feature_extractor = FeatureExtractor(algo=algo)
            self.feature_extractor.load_features(feature_path)
            # 更新检索引擎
            self.retrieval_engine = RetrievalEngine(self.feature_extractor.features)
            return True
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载特征失败: {str(e)}")
            return False

    # ----------------- 核心逻辑 -----------------
    def copy_metrics(self):
        """复制完整的性能报告"""
        text = f"=== 检索性能报告 ===\n"
        text += f"算法类型: {self.feature_algo.currentText()}\n"
        text += f"查询图片: {os.path.basename(self.current_image_path) if self.current_image_path else '未知'}\n\n"

        text += f"=== 测试集统计 ===\n"
        text += f"测试集总图片: {self.test_total_images.text()}\n"
        text += f"测试集类别数: {self.test_total_classes.text()}\n"
        text += f"平均每类图片: {self.test_avg_per_class.text()}\n\n"

        text += f"=== 检索性能 ===\n"
        text += f"总相关图片: {self.metric_calculator.total_relevant}\n"
        text += f"当前召回率: {self.recall.text()}\n"
        text += f"当前精确率: {self.precision.text()}\n"
        text += f"当前mAP@10: {self.map.text()}\n"
        text += f"响应时间: {self.response_time.text()}"

        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        QMessageBox.information(self, "已复制", "性能数据已复制到剪贴板")

    def on_algorithm_changed(self):
        """算法切换事件处理"""
        # 算法切换时不立即加载特征，而是更新显示
        algo = self.feature_algo.currentText()
        self.current_algorithm.setText(algo)
        self.statusBar().showMessage(f"已切换到{algo}算法", 3000)

    def upload_image(self):
        """上传图片并显示预览"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", config.TEST_ROOT, "Images (*.png *.jpg *.bmp)"
        )
        if not file_path:
            return

        self.current_image_path = os.path.normpath(os.path.abspath(file_path)).lower()
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

        # 确保当前算法可用
        algo = self.feature_algo.currentText()
        if not self.ensure_algorithm_support(algo):
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
        self.total_relevant.setText(f" {self.metric_calculator.total_relevant}")

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
        self.pr_canvas.update_plot(
            metrics.get("pr_data", []), metrics.get("interpolated_pr_data", [])
        )

    def analyze_test_set(self):
        """统计测试集信息"""
        test_dir = config.TEST_ROOT
        total_images = 0
        class_counts = {}

        for root, _, files in os.walk(test_dir):
            class_name = os.path.basename(root)
            if class_name == os.path.basename(test_dir):
                continue  # 跳过根目录

            image_count = sum(
                1 for f in files if f.lower().endswith(tuple(config.ALLOWED_EXTENSIONS))
            )
            if image_count > 0:
                class_counts[class_name] = image_count
                total_images += image_count

        total_classes = len(class_counts)
        avg_per_class = total_images / total_classes if total_classes > 0 else 0

        # 更新UI
        self.test_total_images.setText(str(total_images))
        self.test_total_classes.setText(str(total_classes))
        self.test_avg_per_class.setText(f"{avg_per_class:.2f}")

        # 显示详细统计信息
        details = "\n".join([f"{cls}: {count}" for cls, count in class_counts.items()])
        QMessageBox.information(
            self,
            "测试集统计",
            f"总图片数: {total_images}\n"
            f"总类别数: {total_classes}\n"
            f"平均每类图片数: {avg_per_class:.2f}\n\n"
            f"各类别详细统计:\n{details}",
        )


# ======================== 启动程序 ========================
if __name__ == "__main__":
    # 高DPI适配
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)

    # 启动主窗口
    window = ImageRetrievalUI()
    window.show()
    sys.exit(app.exec_())
