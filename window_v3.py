from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton

class ProIllumination(QMainWindow):
    def __init__(self, args):
        super().__init__()

        self.setGeometry(100, 100, 1280, 720)
        self.setWindowTitle("ProIllumination")

        # 参数
        self.args = args

        # 创建主窗口的主部件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # 创建布局管理器
        main_layout = QHBoxLayout(central_widget)

        # 左侧功能选择部分
        left_layout = QVBoxLayout()
        button_labels = ["阴影检测", "阴影去除", "阴影交互", "光照估计"]

        for label in button_labels:
            button = QPushButton(label)
            left_layout.addWidget(button)

        left_layout.setContentsMargins(0, 0, 10, 0)  # 设置左侧布局右侧边距

        # 中间可视化窗口
        visualization_layout = QVBoxLayout()

        # 右侧功能栏
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(10, 0, 0, 0)  # 设置右侧布局左侧边距

        # 将左、中、右布局添加到主布局
        main_layout.addLayout(left_layout)
        main_layout.addLayout(visualization_layout, 1)  # 中间布局占用剩余可用宽度
        main_layout.addLayout(right_layout)

# 测试用
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = ProIllumination(None)
    window.show()
    sys.exit(app.exec_())
