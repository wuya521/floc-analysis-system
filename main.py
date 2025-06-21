import sys
import os
import time
import glob
import cv2
import numpy as np
import pandas as pd
import torch
from math import pi, log10, sqrt
from scipy.spatial import distance
from scipy import stats
from tqdm import tqdm
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
                             QProgressBar, QTextEdit, QSpinBox, QDoubleSpinBox, QTableView,
                             QMessageBox, QComboBox, QGroupBox, QRadioButton, QScrollArea,
                             QSplitter, QGridLayout, QCheckBox, QHeaderView, QSlider,
                             QListWidget, QListWidgetItem, QAbstractItemView, QToolButton,
                             QTableWidget, QTableWidgetItem, QDialog, QProgressDialog,
                             QFormLayout, QStatusBar, QShortcut, QAction, QToolBar)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QIcon, QBrush, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QAbstractTableModel, QPoint, QUrl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from ultralytics import YOLO
import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from io import BytesIO
import io
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from PyQt5.QtGui import QDoubleValidator
from PIL import Image, ImageDraw, ImageFont
# 添加QtWebEngineWidgets导入
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
    WEB_ENGINE_AVAILABLE = True
except ImportError:
    print("QtWebEngineWidgets not available, help system will use external browser")
    WEB_ENGINE_AVAILABLE = False

# 解决中文显示问题 - 修改字体配置，增加更多备选
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 用于替代上标²的字符
SQUARE_SYMBOL = ' sq'

# 检查是否有中文字体
fonts = [f.name for f in font_manager.fontManager.ttflist]
chinese_fonts = [f for f in fonts if '黑体' in f or '雅黑' in f or '宋体' in f or 'SimHei' in f or 'SimSun' in f]
if chinese_fonts:
    plt.rcParams['font.sans-serif'] = chinese_fonts + plt.rcParams['font.sans-serif']

# 样式表 - 粉色主题
DARK_STYLE = """
    QMainWindow, QDialog, QWidget {
        background-color: #FFF5F7;
        color: #333333;
    }
    QPushButton {
        background-color: #FF80AB;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #FF4081;
    }
    QPushButton:pressed {
        background-color: #F50057;
    }
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background-color: #FFFFFF;
        color: #333333;
        border: 1px solid #FFCDD2;
        border-radius: 3px;
        padding: 4px;
        selection-background-color: #FF80AB;
    }
    QTabWidget::pane {
        border: 1px solid #FFCDD2;
        background-color: #FFF5F7;
    }
    QTabBar::tab {
        background-color: #FFCDD2;
        color: #333333;
        padding: 8px 16px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }
    QTabBar::tab:selected {
        background-color: #FF80AB;
        color: white;
    }
    QTabBar::tab:hover:!selected {
        background-color: #FFEBEE;
    }
    QProgressBar {
        border: 1px solid #FFCDD2;
        border-radius: 3px;
        background-color: #FFFFFF;
        color: #333333;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #FF80AB;
        width: 10px;
    }
    QGroupBox {
        border: 1px solid #FFCDD2;
        border-radius: 3px;
        margin-top: 12px;
        font-weight: bold;
        color: #D81B60;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 5px;
        color: #D81B60;
    }
    QLabel {
        color: #333333;
    }
    QTableView {
        background-color: #FFFFFF;
        alternate-background-color: #FFEBEE;
        selection-background-color: #FF80AB;
        border: 1px solid #FFCDD2;
    }
    QHeaderView::section {
        background-color: #FFCDD2;
        color: #333333;
        padding: 4px;
        border: 1px solid #FFF5F7;
    }
    QTextEdit {
        background-color: #FFFFFF;
        color: #333333;
        border: 1px solid #FFCDD2;
    }
    QScrollBar:vertical {
        border: none;
        background: #FFEBEE;
        width: 10px;
        margin: 0px;
    }
    QScrollBar::handle:vertical {
        background: #FFCDD2;
        min-height: 20px;
        border-radius: 5px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    QSlider::groove:horizontal {
        border: 1px solid #FFCDD2;
        height: 8px;
        background: #FFEBEE;
        margin: 2px 0;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: #FF80AB;
        border: 1px solid #FF80AB;
        width: 18px;
        margin: -2px 0;
        border-radius: 9px;
    }
    QCheckBox {
        color: #333333;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
    }
    QCheckBox::indicator:unchecked {
        border: 1px solid #FFCDD2;
        background-color: #FFFFFF;
        border-radius: 3px;
    }
    QCheckBox::indicator:checked {
        border: 1px solid #FF80AB;
        background-color: #FF80AB;
        border-radius: 3px;
        image: url(check.png);
    }
"""

# 内嵌帮助HTML资源
HELP_HTML_CONTENT = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>絮团分析系统使用教程</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f8f8;
        }
        header {
            background-color: #D81B60;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        h1, h2, h3 {
            color: #D81B60;
        }
        .section {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 15px 0;
        }
        .step {
            margin-bottom: 20px;
            padding-left: 20px;
            border-left: 4px solid #FF80AB;
        }
        .note {
            background-color: #FFF9C4;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #F8BBD0;
        }
        code {
            background-color: #f1f1f1;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: Consolas, monospace;
        }
        footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            color: #888;
            font-size: 0.9em;
        }
        .tab-illustration {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
            margin: 20px 0;
        }
        .tab-illustration div {
            flex: 1;
            min-width: 300px;
            background-color: #FCE4EC;
            padding: 15px;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <header>
        <h1>絮团分析系统使用教程</h1>
        <p style="font-size: 1.8em; font-weight: bold; margin: 15px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">基于YOLOv8的絮团检测、测量与分析软件</p>
    </header>

    <div class="section">
        <h2>1. 软件简介</h2>
        <p>絮团分析系统是一款专门用于分析絮团（絮状物质）图像的软件，基于YOLOv8深度学习模型进行絮团检测和分割，帮助研究人员分析絮团的各种形态特性。</p>

        <h3>主要功能：</h3>
        <ul>
            <li>使用深度学习模型自动识别和分割图像中的絮团</li>
            <li>计算絮团多种形态学参数（面积、周长、直径、圆度等）</li>
            <li>生成絮团特性统计图表和数据分析</li>
            <li>支持批量处理大量图像数据</li>
            <li>提供报告生成功能，快速输出分析结果报告</li>
            <li>图像可视化与导出功能</li>
        </ul>
    </div>

    <div class="section">
        <h2>2. 软件界面</h2>
        <p>软件界面由五个主要标签页组成，分别是：处理、可视化、统计、报告生成和批量处理。</p>

        <div class="tab-illustration">
            <div>
                <h3>处理标签页</h3>
                <p>设置模型、输入输出路径，以及处理参数和启动图像处理的主界面。</p>
            </div>
            <div>
                <h3>可视化标签页</h3>
                <p>查看处理后的图像结果，支持缩放、浏览和导出图像。</p>
            </div>
            <div>
                <h3>统计标签页</h3>
                <p>显示处理结果的统计信息和图表，提供数据分析功能。</p>
            </div>
            <div>
                <h3>报告生成标签页</h3>
                <p>基于处理结果生成PDF格式的报告，可自定义报告内容。</p>
            </div>
            <div>
                <h3>批量处理标签页</h3>
                <p>设置多个处理任务，批量处理多个图像文件夹。</p>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>3. 单次图像处理</h2>

        <div class="step">
            <h3>步骤1：设置模型和路径</h3>
            <p>在"处理"标签页中设置以下参数：</p>
            <ul>
                <li><strong>模型路径</strong>：选择YOLOv8分割模型文件（.pt格式）</li>
                <li><strong>图像文件夹</strong>：选择包含待处理图像的文件夹</li>
                <li><strong>输出CSV文件</strong>：设置结果保存的CSV文件路径</li>
                <li><strong>输出图像文件夹</strong>：设置处理后图像的保存位置</li>
            </ul>
            <p>点击各输入框旁的"浏览..."按钮可以打开文件对话框选择路径。</p>
        </div>

        <div class="step">
            <h3>步骤2：设置处理参数</h3>
            <p>设置以下处理参数：</p>
            <ul>
                <li><strong>像素尺寸(微米)</strong>：设置图像中每个像素代表的实际尺寸，影响所有物理测量值</li>
                <li><strong>置信度阈值</strong>：通过滑块设置模型识别的置信度阈值（0-1之间），值越高要求识别越确定</li>
                <li><strong>最小面积阈值(微米²)</strong>：过滤掉面积小于此值的絮团</li>
                <li><strong>最小周长阈值(微米)</strong>：过滤掉周长小于此值的絮团</li>
                <li><strong>保存处理图像</strong>：勾选此项将保存处理后的图像，用于后续可视化</li>
            </ul>
        </div>

        <div class="step">
            <h3>步骤3：开始处理</h3>
            <p>点击"开始处理"按钮启动图像处理过程。处理日志将显示在下方文本框中，进度条会显示当前处理进度。</p>
            <p>处理过程中可以随时点击"停止处理"按钮中断处理。</p>
        </div>

        <div class="note">
            <p><strong>注意</strong>：处理过程中会自动将结果保存到指定的CSV文件中，并在处理完成后自动切换到统计标签页查看结果。</p>
        </div>
    </div>

    <div class="section">
        <h2>4. 结果可视化</h2>

        <div class="step">
            <h3>查看处理图像</h3>
            <p>处理完成后，可以在"可视化"标签页查看处理结果图像：</p>
            <ul>
                <li>使用<strong>图像选择器</strong>下拉列表选择要查看的图像</li>
                <li>使用<strong>上一张</strong>和<strong>下一张</strong>按钮在图像间导航</li>
                <li>使用<strong>缩放</strong>控件调整图像显示大小</li>
                <li>点击<strong>导出当前图像</strong>按钮保存当前显示的图像</li>
            </ul>
        </div>

        <div class="step">
            <h3>显示选项</h3>
            <p>可以设置处理图像的显示选项，控制哪些信息在图像上显示：</p>
            <ul>
                <li><strong>显示轮廓</strong>：在图像上显示检测到的絮团轮廓</li>
                <li><strong>显示标签</strong>：在絮团旁显示编号标签</li>
                <li><strong>显示数据</strong>：在图像上显示主要测量数据</li>
                <li><strong>显示凸包</strong>：显示絮团的凸包轮廓</li>
            </ul>
        </div>
    </div>

    <div class="section">
        <h2>5. 数据统计与分析</h2>

        <div class="step">
            <h3>数据表格</h3>
            <p>在"统计"标签页中，可以查看所有处理结果的数据表格，包含以下信息：</p>
            <table>
                <tr>
                    <th>参数</th>
                    <th>描述</th>
                </tr>
                <tr>
                    <td>ImageName</td>
                    <td>图像文件名</td>
                </tr>
                <tr>
                    <td>FlocID</td>
                    <td>絮团标识符</td>
                </tr>
                <tr>
                    <td>Area_um2</td>
                    <td>絮团面积（微米²）</td>
                </tr>
                <tr>
                    <td>Perimeter_um</td>
                    <td>絮团周长（微米）</td>
                </tr>
                <tr>
                    <td>EquivDiameter_um</td>
                    <td>等效直径（微米）</td>
                </tr>
                <tr>
                    <td>Circularity</td>
                    <td>圆度（0-1，1表示完美圆形）</td>
                </tr>
                <tr>
                    <td>MaxFeretDiameter_um</td>
                    <td>最大Feret直径（微米）</td>
                </tr>
                <tr>
                    <td>MinFeretDiameter_um</td>
                    <td>最小Feret直径（微米）</td>
                </tr>
                <tr>
                    <td>FeretAngle</td>
                    <td>Feret角度</td>
                </tr>
                <tr>
                    <td>AspectRatio</td>
                    <td>纵横比（长宽比）</td>
                </tr>
                <tr>
                    <td>Nf2</td>
                    <td>二维分形维数</td>
                </tr>
                <tr>
                    <td>Nf3</td>
                    <td>三维分形维数</td>
                </tr>
                <tr>
                    <td>Convexity</td>
                    <td>凸性</td>
                </tr>
                <tr>
                    <td>Compactness</td>
                    <td>紧凑度</td>
                </tr>
                <tr>
                    <td>Roughness</td>
                    <td>粗糙度指数</td>
                </tr>
            </table>
        </div>

        <div class="step">
            <h3>统计图表</h3>
            <p>软件自动生成多种统计图表，便于数据分析：</p>
            <ul>
                <li><strong>面积分布直方图</strong>：显示絮团面积的分布情况</li>
                <li><strong>直径分布直方图</strong>：显示絮团等效直径的分布</li>
                <li><strong>圆度分布图</strong>：显示絮团圆度的分布</li>
                <li><strong>大小-形状关系图</strong>：分析絮团大小与形状之间的关系</li>
                <li><strong>分形维数分析图</strong>：显示分形维数的分布</li>
                <li><strong>箱线图</strong>：显示各主要参数的统计分布</li>
            </ul>
            <p>可以点击切换不同类型的图表，查看不同的数据分析视图。</p>
        </div>

        <div class="step">
            <h3>导出图表</h3>
            <p>点击"导出当前图表"按钮可以将当前显示的统计图表保存为图像文件（PNG或JPG格式）。</p>
        </div>
    </div>

    <div class="section">
        <h2>6. 生成报告</h2>

        <div class="step">
            <h3>报告设置</h3>
            <p>在"报告生成"标签页中，可以设置报告的内容和格式：</p>
            <ul>
                <li><strong>报告标题</strong>：设置报告的主标题</li>
                <li><strong>报告副标题</strong>：设置报告的副标题</li>
                <li><strong>数据来源</strong>：选择使用当前处理结果或从文件加载数据</li>
                <li><strong>选择图表</strong>：勾选要包含在报告中的统计图表</li>
                <li><strong>包含统计表格</strong>：是否在报告中包含统计数据表格</li>
                <li><strong>包含原始数据</strong>：是否在报告中包含原始数据表格</li>
                <li><strong>报告路径</strong>：设置报告保存的PDF文件路径</li>
            </ul>
        </div>

        <div class="step">
            <h3>预览和生成报告</h3>
            <p>设置完成后，可以：</p>
            <ul>
                <li>点击<strong>预览报告</strong>按钮查看报告预览</li>
                <li>点击<strong>生成报告</strong>按钮生成PDF报告并保存到指定位置</li>
            </ul>
            <p>生成的报告包含您选择的所有内容，结构清晰，便于分享和发布。</p>
        </div>
    </div>

    <div class="section">
        <h2>7. 批量处理</h2>

        <div class="step">
            <h3>添加批处理任务</h3>
            <p>在"批量处理"标签页中，可以设置多个处理任务，批量处理大量图像：</p>
            <ol>
                <li>点击<strong>添加任务</strong>按钮打开任务设置对话框</li>
                <li>设置<strong>任务名称</strong>（便于识别）</li>
                <li>设置<strong>输入图像文件夹</strong>（包含待处理图像的文件夹）</li>
                <li>设置<strong>输出CSV文件</strong>（结果数据保存位置）</li>
                <li>设置<strong>输出图像文件夹</strong>（处理后图像保存位置）</li>
                <li>点击<strong>确定</strong>添加任务到列表</li>
            </ol>
            <p>可以添加多个任务，它们将按顺序执行。</p>
        </div>

        <div class="step">
            <h3>批处理设置</h3>
            <p>设置所有批处理任务共用的参数：</p>
            <ul>
                <li><strong>模型路径</strong>：选择用于所有任务的YOLOv8模型</li>
                <li><strong>置信度阈值</strong>：设置模型识别的置信度阈值</li>
                <li><strong>像素尺寸(μm)</strong>：设置图像的像素物理尺寸</li>
                <li><strong>保存处理后的图像</strong>：是否保存处理结果图像</li>
            </ul>
        </div>

        <div class="step">
            <h3>开始批处理</h3>
            <p>添加任务并设置参数后，点击<strong>开始批量处理</strong>按钮启动批处理过程。</p>
            <p>系统会依次处理每个任务，在进度区域显示总体进度和当前任务进度，处理日志会实时更新。</p>
            <p>可以随时点击<strong>停止批量处理</strong>按钮中断处理过程。</p>
        </div>

        <div class="note">
            <p><strong>提示</strong>：批处理完成后，每个任务的结果会分别保存到各自设置的CSV文件中。您可以在报告生成标签页中单独加载这些文件进行报告生成。</p>
        </div>
    </div>

    <div class="section">
        <h2>8. 高级功能与技巧</h2>

        <div class="step">
            <h3>调整置信度阈值</h3>
            <p>置信度阈值是模型识别絮团的关键参数：</p>
            <ul>
                <li><strong>较低的阈值</strong>（0.2-0.4）：可以检测到更多可能的絮团，但可能包含一些错误识别</li>
                <li><strong>中等阈值</strong>（0.4-0.6）：平衡检测率和准确性的推荐值</li>
                <li><strong>较高的阈值</strong>（0.6-0.8）：只检测模型非常确定的絮团，减少误识别但可能漏检一些絮团</li>
            </ul>
            <p>根据图像质量和絮团特性调整阈值，获得最佳检测效果。</p>
        </div>

        <div class="step">
            <h3>过滤参数优化</h3>
            <p>通过调整过滤参数可以排除不符合要求的絮团：</p>
            <ul>
                <li>增加<strong>最小面积阈值</strong>可以排除小颗粒或噪点</li>
                <li>增加<strong>最小周长阈值</strong>可以排除形状过于简单的检测结果</li>
            </ul>
            <p>根据研究需求设置合适的过滤参数，确保分析结果的准确性。</p>
        </div>

        <div class="step">
            <h3>数据分析技巧</h3>
            <p>使用统计标签页的数据和图表进行高级分析：</p>
            <ul>
                <li>观察<strong>面积-圆度散点图</strong>可以发现絮团大小与形状的关系</li>
                <li>使用<strong>分形维数</strong>分析絮团的复杂性和结构特性</li>
                <li>比较不同样本的<strong>箱线图</strong>可以快速识别样本间的差异</li>
            </ul>
        </div>
    </div>

    <div class="section">
        <h2>9. 常见问题解答</h2>

        <div class="step">
            <h3>软件卡顿或崩溃</h3>
            <p><strong>问题</strong>：处理大量高分辨率图像时软件运行缓慢或崩溃。</p>
            <p><strong>解决方案</strong>：</p>
            <ul>
                <li>将数据分成多个较小的批次进行处理</li>
                <li>确保计算机有足够的内存（推荐8GB以上）</li>
                <li>关闭其他占用资源的程序</li>
                <li>处理非常大的图像时，考虑预先降低图像分辨率</li>
            </ul>
        </div>

        <div class="step">
            <h3>模型识别效果不佳</h3>
            <p><strong>问题</strong>：模型对絮团的识别率低或有大量误识别。</p>
            <p><strong>解决方案</strong>：</p>
            <ul>
                <li>调整置信度阈值找到最佳平衡点</li>
                <li>确保使用的模型与您的絮团类型匹配</li>
                <li>改善图像质量和对比度</li>
                <li>联系开发者获取针对特定絮团类型的优化模型</li>
            </ul>
        </div>

        <div class="step">
            <h3>如何处理不同放大倍率的图像</h3>
            <p><strong>问题</strong>：需要处理不同显微镜放大倍率拍摄的图像。</p>
            <p><strong>解决方案</strong>：</p>
            <ul>
                <li>为每种放大倍率创建单独的批处理任务</li>
                <li>为每个任务设置正确的像素尺寸值</li>
                <li>处理完成后可以将多个结果文件合并分析</li>
            </ul>
        </div>
    </div>

    <footer>
        <p>本软件由 Ya Wu 开发，训练数据由 Ya Wu 和 Ying Chen 共同标注</p>
        <p>版权所有 © 2025 - 保留所有权利</p>
    </footer>
</body>
</html>"""

def write_help_file():
    """将内嵌的帮助HTML内容写入文件"""
    help_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "teaching.html")
    try:
        with open(help_path, 'w', encoding='utf-8') as f:
            f.write(HELP_HTML_CONTENT)
        return True
    except Exception as e:
        print(f"Error writing help file: {e}")
        return False


class PandasModel(QAbstractTableModel):
    """表格数据模型，用于显示DataFrame数据"""

    def __init__(self, data):
        super(PandasModel, self).__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])
        return None


class ProcessThread(QThread):
    """处理图像的后台线程"""
    progress_signal = pyqtSignal(int, str)
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(pd.DataFrame)
    image_signal = pyqtSignal(QPixmap)
    stats_signal = pyqtSignal(dict)

    def __init__(self, model_path, image_folder, output_csv, output_image_folder,
                 pixel_size_um, confidence_threshold, save_images=True,
                 min_area=3.0, min_perimeter=3.0, min_circularity=0.0, max_aspect_ratio=10.0,
                 edge_detection_mode="strict", parent=None):
        QThread.__init__(self, parent)
        self.model_path = model_path
        self.image_folder = image_folder
        self.output_csv = output_csv
        self.output_image_folder = output_image_folder
        self.pixel_size_um = pixel_size_um
        self.confidence_threshold = confidence_threshold
        self.save_images = save_images
        self.min_area = min_area
        self.min_perimeter = min_perimeter
        self.min_circularity = min_circularity
        self.max_aspect_ratio = max_aspect_ratio
        self.edge_detection_mode = edge_detection_mode
        self.running = True

    def calculate_floc_properties(self, mask, pixel_size_um, img_shape=None):
        """计算絮团的形态属性，将像素单位转换为微米"""
        try:
            # 确保掩码是二维的，并且格式正确
            if len(mask.shape) > 2:
                mask = mask[0] if mask.shape[0] == 1 else mask[:, :, 0]

            # 确保掩码是二值图像
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)

            # 检查掩码是否只包含0和1
            if np.max(mask) > 1:
                mask = (mask > 0).astype(np.uint8)

            # 查找轮廓 - 用于周长计算
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if not contours or len(contours) == 0:
                return None

            # 选择最大的轮廓
            contour = max(contours, key=cv2.contourArea)

            # 计算面积（像素和微米）
            area_px = cv2.contourArea(contour)
            area_um2 = area_px * (pixel_size_um ** 2)

            # 检查面积是否满足最小要求（使用微米单位）
            if area_um2 < self.min_area:
                return None

            # 计算周长（像素和微米）
            perimeter_px = cv2.arcLength(contour, True)
            perimeter_um = perimeter_px * pixel_size_um

            # 检查周长是否满足最小要求（使用微米单位）
            if perimeter_um < self.min_perimeter:
                return None

            xy = contour.reshape(-1, 2)  # 重塑为Nx2点数组

            # 检查絮团是否在图像边缘
            if img_shape is not None:
                height, width = img_shape

                # 边缘检测模式
                if self.edge_detection_mode == "strict":
                    # 严格模式：任何点在边缘就跳过
                    for point in xy:
                        x, y = point
                        if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                            return None  # 絮团在边缘，跳过
                else:
                    # 宽松模式：只检查中心点是否在边缘
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        if cx == 0 or cy == 0 or cx == width - 1 or cy == height - 1:
                            return None  # 中心在边缘，跳过

            # 1. 面积（微米²）
            area_px = cv2.contourArea(contour)
            area_um2 = area_px * (pixel_size_um ** 2)

            # 2. 周长（微米）
            perimeter_px = cv2.arcLength(contour, True)
            perimeter_um = perimeter_px * pixel_size_um

            # 3. 等效直径（微米）
            equiv_diameter_um = 2 * np.sqrt(area_um2 / pi)

            # 4. 圆度
            circularity = 4 * pi * area_px / (perimeter_px ** 2) if perimeter_px > 0 else 0

            # 检查圆度是否满足最小要求
            if circularity < self.min_circularity:
                return None

            # 计算质心
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            # 替代圆度计算（基于距离）
            points = np.array(contour).reshape(-1, 2)
            center = np.array([cx, cy])
            if len(points) > 0:
                distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
                mean_distance = np.mean(distances)
                if mean_distance > 0:
                    circularity_alt = 1 - (np.sqrt(np.mean((distances - mean_distance) ** 2)) / mean_distance)
                else:
                    circularity_alt = 0
            else:
                circularity_alt = 0

            # 5. Feret直径和角度
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # 计算边界框对角线距离
            d1 = distance.euclidean(box[0], box[2])
            d2 = distance.euclidean(box[1], box[3])

            max_feret_diameter_px = max(d1, d2)
            min_feret_diameter_px = min(d1, d2)

            # 转换为微米
            max_feret_diameter_um = max_feret_diameter_px * pixel_size_um
            min_feret_diameter_um = min_feret_diameter_px * pixel_size_um

            # 计算最大Feret直径的角度
            angle = rect[2]
            if d1 < d2:  # 如果宽度<高度，调整角度
                angle = angle + 90 if angle < 0 else angle - 90

            # 6. 二维分形维数 (Nf2)
            if area_px > 0 and perimeter_px > 0:
                nf2 = 2 * log10(perimeter_px) / log10(area_px)
            else:
                nf2 = 0

            # 7. 三维分形维数 (Nf3) - 简化近似
            if nf2 > 0:
                # 基于Nf2的简化Nf3近似
                nf3 = 2.0 + (nf2 - 1.0) * 0.5  # 基于文献关系的简单缩放
            else:
                nf3 = 0

            # 8. 计算纵横比（长宽比）
            aspect_ratio = max_feret_diameter_um / min_feret_diameter_um if min_feret_diameter_um > 0 else 1.0

            # 检查纵横比是否超过最大限制
            if aspect_ratio > self.max_aspect_ratio:
                return None

            # 9. 计算凸性（Convexity）
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area_px / hull_area if hull_area > 0 else 0

            # 10. 计算紧凑度（Compactness）
            compactness = np.sqrt(4 * area_px / pi) / max_feret_diameter_px if max_feret_diameter_px > 0 else 0

            # 11. 粗糙度指数（Roughness）
            hull_perimeter = cv2.arcLength(hull, True)
            roughness = perimeter_px / hull_perimeter if hull_perimeter > 0 else 1.0

            # 12. 保存轮廓信息用于可视化
            contour_info = {
                'contour': contour,
                'center': (cx, cy),
                'area': area_px,
                'hull': hull
            }

            # 返回所有计算的属性
            return {
                'Area_um2': area_um2,
                'Perimeter_um': perimeter_um,
                'EquivDiameter_um': equiv_diameter_um,
                'Circularity': circularity,
                'Circularity_alt': circularity_alt,
                'MaxFeretDiameter_um': max_feret_diameter_um,
                'MinFeretDiameter_um': min_feret_diameter_um,
                'FeretAngle': angle,
                'AspectRatio': aspect_ratio,
                'Convexity': convexity,
                'Compactness': compactness,
                'Roughness': roughness,
                'Nf2': nf2,
                'Nf3': nf3,
                '_contour_info': contour_info  # 仅用于内部可视化
            }
        except Exception as e:
            self.log_signal.emit(f"处理掩码时出错: {e}")
            return None

    def process_current_image(self, image_path, model, img_idx, total_images, stats):
        """处理单个图像"""
        try:
            # 加载图像并运行推理
            image = cv2.imread(image_path)
            if image is None:
                self.log_signal.emit(f"错误：无法读取图像 {image_path}")
                return [], None

            img_height, img_width = image.shape[:2]
            image_name = os.path.basename(image_path)

            # 运行推理
            results = model(image, verbose=False)

            # 处理结果变量
            processed_flocs = []
            edge_flocs_in_image = 0
            small_flocs_in_image = 0
            low_conf_flocs_in_image = 0
            error_flocs_in_image = 0

            # 可视化图像
            vis_image = image.copy()

            # 更新状态
            self.progress_signal.emit(img_idx * 100 // total_images,
                                      f"处理图像: {image_name} ({img_idx + 1}/{total_images})")

            detected_count = 0
            masks_count = 0

            for result in results:
                # 检查是否有掩码（分割）和置信度
                if result.masks is not None and result.boxes is not None:
                    masks = result.masks.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()

                    detected_count = len(masks)
                    stats['total_detected'] += detected_count

                    valid_masks = []
                    valid_confidences = []

                    # 过滤低置信度的检测
                    for i, (mask_data, conf) in enumerate(zip(masks, confidences)):
                        if conf >= self.confidence_threshold:
                            valid_masks.append(mask_data)
                            valid_confidences.append(conf)
                        else:
                            low_conf_flocs_in_image += 1
                            stats['skipped_low_conf'] += 1

                    masks = valid_masks
                    confidences = valid_confidences
                    masks_count = len(masks)
                    stats['total_valid'] += masks_count

                    # 处理每个检测到的絮团
                    for i, (mask_data, conf) in enumerate(zip(masks, confidences)):
                        if not self.running:
                            return [], None

                        try:
                            # 获取掩码
                            mask = mask_data.data

                            # 检查掩码形状和类型
                            if mask is None or mask.size == 0:
                                error_flocs_in_image += 1
                                stats['error_flocs'] += 1
                                continue

                            # 计算絮团属性（传递图像尺寸以检查边缘）
                            props = self.calculate_floc_properties(mask, self.pixel_size_um, (img_height, img_width))

                            if props:
                                # 添加絮团ID、图像索引、图像名称和置信度
                                floc_id = stats['processed_count'] + 1
                                props['FlocID'] = floc_id
                                props['ImageNumber'] = img_idx + 1
                                props['ImageName'] = image_name
                                props['Confidence'] = round(float(conf), 2)

                                # 绘制有效絮团（绿色）
                                contour_info = props.pop('_contour_info', None)
                                if contour_info:
                                    cv2.drawContours(vis_image, [contour_info['contour']], 0, (0, 255, 0), 2)
                                    # 可选绘制凸包
                                    cv2.drawContours(vis_image, [contour_info['hull']], 0, (0, 200, 0), 1)
                                    cx, cy = contour_info['center']
                                    cv2.putText(vis_image, f"{floc_id}: {conf:.2f}", (cx, cy),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                                # 添加到结果列表
                                processed_flocs.append(props)
                                stats['processed_count'] += 1
                            else:
                                # 尝试判断是边缘还是太小
                                temp_mask = mask.copy()
                                if len(temp_mask.shape) > 2:
                                    temp_mask = temp_mask[0] if temp_mask.shape[0] == 1 else temp_mask[:, :, 0]
                                if temp_mask.dtype != np.uint8:
                                    temp_mask = temp_mask.astype(np.uint8)
                                if np.max(temp_mask) > 1:
                                    temp_mask = (temp_mask > 0).astype(np.uint8)

                                contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                                if not contours or len(contours) == 0:
                                    error_flocs_in_image += 1
                                    stats['error_flocs'] += 1
                                    continue

                                contour = max(contours, key=cv2.contourArea)
                                area = cv2.contourArea(contour)

                                # 检查是否在边缘
                                xy = contour.reshape(-1, 2)
                                is_edge = False
                                for point in xy:
                                    x, y = point
                                    if x == 0 or y == 0 or x == img_width - 1 or y == img_height - 1:
                                        is_edge = True
                                        break

                                if is_edge:
                                    # 绘制边缘絮团（红色）
                                    cv2.drawContours(vis_image, [contour], 0, (0, 0, 255), 2)
                                    edge_flocs_in_image += 1
                                    stats['skipped_edge'] += 1
                                elif area * (self.pixel_size_um ** 2) < self.min_area:
                                    # 绘制过小絮团（黄色）
                                    cv2.drawContours(vis_image, [contour], 0, (0, 255, 255), 2)
                                    small_flocs_in_image += 1
                                    stats['skipped_small'] += 1
                                else:
                                    error_flocs_in_image += 1
                                    stats['error_flocs'] += 1

                        except Exception as e:
                            self.log_signal.emit(f"处理图像 {image_name} 中的絮团 {i + 1} 时出错: {e}")
                            error_flocs_in_image += 1
                            stats['error_flocs'] += 1

            # 添加统计信息到图像 - 使用PIL绘制中文
            # 将OpenCV图像转换为PIL图像
            vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(vis_image_rgb)
            draw = ImageDraw.Draw(pil_img)

            # 尝试加载中文字体，如果失败则使用默认字体
            try:
                # 尝试加载系统中的中文字体
                font_path = None
                for font_name in ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong']:
                    for font in font_manager.fontManager.ttflist:
                        if font_name.lower() in font.name.lower():
                            font_path = font.fname
                            break
                    if font_path:
                        break

                if not font_path:
                    # 如果找不到中文字体，尝试使用系统默认字体
                    font_path = font_manager.findfont(font_manager.FontProperties())

                # 创建字体对象
                font = ImageFont.truetype(font_path, 30)
            except Exception as e:
                self.log_signal.emit(f"加载字体出错: {e}，使用默认字体")
                font = ImageFont.load_default()

            # 绘制文本
            draw.text((10, 10), f"有效: {len(processed_flocs)}", fill=(0, 255, 0), font=font)
            draw.text((10, 50), f"边缘: {edge_flocs_in_image}", fill=(0, 0, 255), font=font)
            draw.text((10, 90), f"过小: {small_flocs_in_image}", fill=(0, 255, 255), font=font)
            draw.text((10, 130), f"低置信度: {low_conf_flocs_in_image}", fill=(255, 0, 255), font=font)

            # 将PIL图像转换回OpenCV格式
            vis_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # 输出处理信息
            self.log_signal.emit(f"图像 {img_idx + 1}/{total_images} ({image_name}): 检测 {detected_count} 个絮团, "
                                 f"置信度>={self.confidence_threshold}: {masks_count}, "
                                 f"处理 {len(processed_flocs)} 个有效絮团, "
                                 f"边缘 {edge_flocs_in_image} 个, "
                                 f"过小 {small_flocs_in_image} 个, "
                                 f"低置信度 {low_conf_flocs_in_image} 个")

            # 保存处理图像
            if self.save_images and self.output_image_folder:
                os.makedirs(self.output_image_folder, exist_ok=True)
                output_path = os.path.join(self.output_image_folder, f"processed_{image_name}")
                cv2.imwrite(output_path, vis_image)

            # 转换为Qt可显示的图像
            height, width, channel = vis_image.shape
            bytesPerLine = 3 * width
            qImg = QImage(vis_image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)

            # 发送图像信号
            self.image_signal.emit(pixmap)

            # 发送统计信号
            self.stats_signal.emit(stats)

            return processed_flocs, vis_image

        except Exception as e:
            self.log_signal.emit(f"处理图像 {image_path} 时出错: {e}")
            return [], None

    def run(self):
        # 初始化统计
        stats = {
            'total_detected': 0,  # 总检测絮团数
            'total_valid': 0,  # 有效置信度絮团数
            'processed_count': 0,  # 处理的絮团数
            'skipped_edge': 0,  # 跳过的边缘絮团
            'skipped_small': 0,  # 跳过的小絮团
            'skipped_low_conf': 0,  # 跳过的低置信度絮团
            'error_flocs': 0  # 处理错误的絮团
        }

        try:
            # 加载模型
            self.log_signal.emit("正在加载模型...")
            model = YOLO(self.model_path)
            if torch.cuda.is_available():
                model.to('cuda')
                self.log_signal.emit(f"使用GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.log_signal.emit("未检测到GPU，使用CPU")

            # 获取图像文件列表
            self.log_signal.emit("正在搜索图像文件...")
            image_files = glob.glob(os.path.join(self.image_folder, "*.jpg")) + \
                          glob.glob(os.path.join(self.image_folder, "*.png")) + \
                          glob.glob(os.path.join(self.image_folder, "*.jpeg"))

            total_images = len(image_files)
            self.log_signal.emit(f"找到 {total_images} 个图像文件")

            if total_images == 0:
                self.log_signal.emit("未找到图像文件")
                return

            # 初始化结果列表
            all_results = []

            # 处理每个图像
            start_time = time.time()

            for img_idx, img_path in enumerate(image_files):
                if not self.running:
                    self.log_signal.emit("处理已取消")
                    return

                # 处理当前图像
                processed_flocs, _ = self.process_current_image(img_path, model, img_idx, total_images, stats)

                # 添加到结果
                all_results.extend(processed_flocs)

            # 转换结果为DataFrame
            if all_results:
                self.log_signal.emit("\n正在生成结果数据...")
                df = pd.DataFrame(all_results)

                # 重新排序列，将FlocID、ImageNumber、ImageName和Confidence放在前面
                cols = ['FlocID', 'ImageNumber', 'ImageName', 'Confidence'] + [col for col in df.columns if
                                                                               col not in ['FlocID', 'ImageNumber',
                                                                                           'ImageName', 'Confidence']]
                df = df[cols]

                # 将浮点数列格式化为两位小数
                float_columns = df.select_dtypes(include=['float']).columns
                for col in float_columns:
                    df[col] = df[col].round(2)

                # 保存为CSV
                self.log_signal.emit(f"正在保存结果到 {self.output_csv}...")
                os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
                df.to_csv(self.output_csv, index=False, float_format='%.2f')
                self.log_signal.emit(f"结果已成功保存！")

                # 输出统计信息
                self.log_signal.emit(f"\n统计信息:")
                self.log_signal.emit(f"总检测絮团数: {stats['total_detected'] + stats['skipped_low_conf']}")
                self.log_signal.emit(f"置信度>={self.confidence_threshold}的絮团数: {stats['total_valid']}")
                self.log_signal.emit(f"置信度<{self.confidence_threshold}的絮团数: {stats['skipped_low_conf']}")
                self.log_signal.emit(f"有效处理絮团数: {stats['processed_count']}")
                self.log_signal.emit(f"边缘被过滤絮团数: {stats['skipped_edge']}")
                self.log_signal.emit(f"过小被过滤絮团数: {stats['skipped_small']}")
                self.log_signal.emit(f"处理错误絮团数: {stats['error_flocs']}")

                end_time = time.time()
                processing_time = end_time - start_time
                self.log_signal.emit(f"\n总结: 已成功处理 {len(df)} 个絮团，来自 {df['ImageNumber'].nunique()} 张图像")
                self.log_signal.emit(f"处理时间: {processing_time:.2f} 秒 ({processing_time / 60:.2f} 分钟)")

                # 发送完成信号
                self.done_signal.emit(df)
            else:
                self.log_signal.emit("在提供的图像中未检测到有效絮团")
                self.done_signal.emit(pd.DataFrame())

        except Exception as e:
            self.log_signal.emit(f"处理过程出错: {e}")
            self.done_signal.emit(pd.DataFrame())

    def stop(self):
        self.running = False


class StatisticsWidget(QWidget):
    """统计信息显示和图表"""

    def __init__(self, parent=None):
        super(StatisticsWidget, self).__init__(parent)

        # 设置粉色系图表样式
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
            '#FF4081', '#F06292', '#EC407A', '#E91E63', '#D81B60',
            '#C2185B', '#AD1457', '#880E4F', '#FF80AB', '#FF80AB'
        ])
        plt.rcParams['figure.facecolor'] = '#FFF5F7'
        plt.rcParams['axes.facecolor'] = '#FFFFFF'
        plt.rcParams['savefig.facecolor'] = '#FFF5F7'
        plt.rcParams['grid.color'] = '#FFCDD2'

        # 主布局
        layout = QVBoxLayout(self)

        # 分割布局 - 上方为图表和控制区域，下方为数据表格
        main_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(main_splitter)

        # 上方区域 - 分为左侧图表和右侧控制面板
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        main_splitter.addWidget(top_widget)

        # 左侧图表区域
        chart_widget = QWidget()
        chart_layout = QHBoxLayout(chart_widget)
        top_layout.addWidget(chart_widget, 3)

        # 上下分割图表区域
        chart_splitter = QSplitter(Qt.Vertical)
        chart_layout.addWidget(chart_splitter)

        # 饼图 - 处理结果分布
        pie_widget = QWidget()
        pie_layout = QVBoxLayout(pie_widget)
        self.pie_figure = Figure(figsize=(5, 4), dpi=100)
        self.pie_canvas = FigureCanvas(self.pie_figure)
        pie_layout.addWidget(self.pie_canvas)
        chart_splitter.addWidget(pie_widget)

        # 主要分析图表
        main_chart_widget = QWidget()
        main_chart_layout = QVBoxLayout(main_chart_widget)
        self.main_figure = Figure(figsize=(5, 4), dpi=100)
        self.main_canvas = FigureCanvas(self.main_figure)
        main_chart_layout.addWidget(self.main_canvas)
        chart_splitter.addWidget(main_chart_widget)

        # 右侧控制面板
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        top_layout.addWidget(control_widget, 1)

        # 图表类型选择组
        chart_type_group = QGroupBox("图表类型")
        chart_type_layout = QVBoxLayout(chart_type_group)
        control_layout.addWidget(chart_type_group)

        # 图表类型选择下拉框
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "面积直方图", "直径直方图", "圆度直方图", "Feret直径直方图",
            "面积/直径散点图", "圆度/直径散点图", "Feret长短径散点图",
            "分形维数/面积散点图", "纵横比/面积散点图", "纵横比直方图",
            "凸性直方图", "紧凑度直方图", "粗糙度直方图",
            "面积箱线图", "参数相关性热图"
        ])
        self.chart_type_combo.currentIndexChanged.connect(self.update_chart)
        chart_type_layout.addWidget(self.chart_type_combo)

        # 参数设置组
        param_group = QGroupBox("参数设置")
        param_layout = QGridLayout(param_group)
        control_layout.addWidget(param_group)

        # 柱状图箱数设置
        param_layout.addWidget(QLabel("直方图箱数:"), 0, 0)
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(5, 100)
        self.bins_spin.setValue(20)
        self.bins_spin.valueChanged.connect(self.update_chart)
        param_layout.addWidget(self.bins_spin, 0, 1)

        # 分组依据（按图像分组等）
        param_layout.addWidget(QLabel("分组依据:"), 1, 0)
        self.group_by_combo = QComboBox()
        self.group_by_combo.addItems(["无", "图像"])
        self.group_by_combo.currentIndexChanged.connect(self.update_chart)
        param_layout.addWidget(self.group_by_combo, 1, 1)

        # 箱线图分组
        param_layout.addWidget(QLabel("箱线图分组:"), 2, 0)
        self.boxplot_combo = QComboBox()
        self.boxplot_combo.addItems(["按图像", "按面积范围", "按圆度范围", "按直径范围"])
        param_layout.addWidget(self.boxplot_combo, 2, 1)

        # 更新图表按钮
        self.update_chart_btn = QPushButton("更新图表")
        self.update_chart_btn.clicked.connect(self.update_chart)
        control_layout.addWidget(self.update_chart_btn)

        # 导出图表按钮
        self.export_chart_btn = QPushButton("导出图表")
        self.export_chart_btn.clicked.connect(self.export_chart)
        control_layout.addWidget(self.export_chart_btn)

        # 数据摘要
        summary_group = QGroupBox("数据摘要")
        summary_layout = QVBoxLayout(summary_group)
        control_layout.addWidget(summary_group)

        # 统计摘要
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(150)
        summary_layout.addWidget(self.summary_text)

        # 下方表格区域
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        main_splitter.addWidget(table_widget)

        # 数据表格
        self.table_view = QTableView()
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(QLabel("絮团数据:"))
        table_layout.addWidget(self.table_view)

        # 设置分割比例
        main_splitter.setSizes([600, 400])
        chart_splitter.setSizes([300, 300])

        # 初始化变量
        self.df = None
        self.stats = None

    def update_statistics(self, df, stats=None):
        """更新统计图表和表格"""
        if df.empty:
            return

        # 保存数据
        self.df = df
        self.stats = stats

        # 清除旧图表
        self.pie_figure.clear()
        self.main_figure.clear()

        # 绘制饼图 - 处理结果分布
        if stats:
            ax1 = self.pie_figure.add_subplot(111)
            labels = ['有效絮团', '边缘絮团', '过小絮团', '低置信度絮团']
            sizes = [stats['processed_count'], stats['skipped_edge'],
                     stats['skipped_small'], stats['skipped_low_conf']]
            colors = ['#FF80AB', '#F06292', '#E91E63', '#C2185B']  # 使用粉色系列
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            ax1.set_title('絮团处理结果分布', color='#D81B60')  # 粉色标题
            self.pie_figure.tight_layout()
            self.pie_canvas.draw()

        # 绘制默认的主图表（面积直方图）
        self.update_chart()

        # 更新摘要文本
        if stats:
            summary = f"总检测絮团数: {stats['total_detected'] + stats['skipped_low_conf']}\n"
            summary += f"有效处理絮团数: {stats['processed_count']}\n"
            summary += f"边缘被过滤絮团数: {stats['skipped_edge']}\n"
            summary += f"过小被过滤絮团数: {stats['skipped_small']}\n"
            summary += f"低置信度絮团数: {stats['skipped_low_conf']}\n"
            summary += f"处理错误絮团数: {stats['error_flocs']}\n\n"

            # 添加关键统计量
            if not df.empty:
                summary += f"絮团面积统计(μm{SQUARE_SYMBOL}):\n"
                summary += f"  平均值: {df['Area_um2'].mean():.2f}\n"
                summary += f"  中位数: {df['Area_um2'].median():.2f}\n"
                summary += f"  最大值: {df['Area_um2'].max():.2f}\n"
                summary += f"  最小值: {df['Area_um2'].min():.2f}\n\n"

                summary += f"絮团直径平均值: {df['EquivDiameter_um'].mean():.2f} μm\n"
                summary += f"圆度平均值: {df['Circularity'].mean():.2f}\n"
                summary += f"纵横比平均值: {df['AspectRatio'].mean():.2f}\n"
                summary += f"分形维数(Nf2)平均值: {df['Nf2'].mean():.2f}"

            self.summary_text.setText(summary)

        # 更新表格
        model = PandasModel(df)
        self.table_view.setModel(model)

    def update_chart(self):
        """根据选择更新主图表"""
        if self.df is None or self.df.empty:
            return

        # 清除主图表
        self.main_figure.clear()
        ax = self.main_figure.add_subplot(111)

        # 设置粉色主题
        ax.set_facecolor('#FFFFFF')
        ax.tick_params(colors='#D81B60')
        ax.spines['bottom'].set_color('#FFCDD2')
        ax.spines['top'].set_color('#FFCDD2')
        ax.spines['left'].set_color('#FFCDD2')
        ax.spines['right'].set_color('#FFCDD2')

        # 获取参数
        chart_type = self.chart_type_combo.currentText()
        bins = self.bins_spin.value()
        group_by = self.group_by_combo.currentText()

        try:
            # 根据图表类型绘制不同的图
            if chart_type == "面积直方图":
                if group_by == "图像":
                    # 按图像分组绘制
                    for name, group in self.df.groupby('ImageName'):
                        ax.hist(group['Area_um2'], bins=bins, alpha=0.5, label=name)
                    ax.legend()
                else:
                    # 不分组
                    ax.hist(self.df['Area_um2'], bins=bins, alpha=0.7, color='#FF80AB')
                ax.set_xlabel(f'絮团面积 (μm{SQUARE_SYMBOL})', color='#D81B60')
                ax.set_ylabel('频数', color='#D81B60')
                ax.set_title('絮团面积分布', color='#D81B60', fontweight='bold')

            elif chart_type == "直径直方图":
                if group_by == "图像":
                    for name, group in self.df.groupby('ImageName'):
                        ax.hist(group['EquivDiameter_um'], bins=bins, alpha=0.5, label=name)
                    ax.legend()
                else:
                    ax.hist(self.df['EquivDiameter_um'], bins=bins, alpha=0.7, color='#4CAF50')
                ax.set_xlabel('等效直径 (μm)')
                ax.set_ylabel('频数')
                ax.set_title('絮团直径分布')

            elif chart_type == "圆度直方图":
                if group_by == "图像":
                    for name, group in self.df.groupby('ImageName'):
                        ax.hist(group['Circularity'], bins=bins, alpha=0.5, label=name)
                    ax.legend()
                else:
                    ax.hist(self.df['Circularity'], bins=bins, alpha=0.7, color='#FFC107')
                ax.set_xlabel('圆度')
                ax.set_ylabel('频数')
                ax.set_title('絮团圆度分布')

            elif chart_type == "Feret直径直方图":
                if group_by == "图像":
                    for name, group in self.df.groupby('ImageName'):
                        ax.hist(group['MaxFeretDiameter_um'], bins=bins, alpha=0.5, label=name)
                    ax.legend()
                else:
                    ax.hist(self.df['MaxFeretDiameter_um'], bins=bins, alpha=0.7, color='#9C27B0')
                ax.set_xlabel('最大Feret直径 (μm)')
                ax.set_ylabel('频数')
                ax.set_title('絮团最大Feret直径分布')

            elif chart_type == "面积/直径散点图":
                ax.scatter(self.df['Area_um2'], self.df['EquivDiameter_um'], alpha=0.7, c='#2196F3')
                ax.set_xlabel(f'面积 (μm{SQUARE_SYMBOL})')
                ax.set_ylabel('等效直径 (μm)')
                ax.set_title('絮团面积与直径关系')

                # 添加趋势线
                z = np.polyfit(self.df['Area_um2'], self.df['EquivDiameter_um'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(self.df['Area_um2'].min(), self.df['Area_um2'].max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8)

            elif chart_type == "圆度/直径散点图":
                ax.scatter(self.df['EquivDiameter_um'], self.df['Circularity'], alpha=0.7, c='#4CAF50')
                ax.set_xlabel('等效直径 (μm)')
                ax.set_ylabel('圆度')
                ax.set_title('絮团直径与圆度关系')

            elif chart_type == "Feret长短径散点图":
                ax.scatter(self.df['MaxFeretDiameter_um'], self.df['MinFeretDiameter_um'], alpha=0.7, c='#FF5722')
                ax.set_xlabel('最大Feret直径 (μm)')
                ax.set_ylabel('最小Feret直径 (μm)')
                ax.set_title('絮团Feret长短径关系')

                # 添加趋势线
                z = np.polyfit(self.df['MaxFeretDiameter_um'], self.df['MinFeretDiameter_um'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(self.df['MaxFeretDiameter_um'].min(), self.df['MaxFeretDiameter_um'].max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8)

            elif chart_type == "分形维数/面积散点图":
                ax.scatter(self.df['Area_um2'], self.df['Nf2'], alpha=0.7, c='#9C27B0')
                ax.set_xlabel(f'面积 (μm{SQUARE_SYMBOL})')
                ax.set_ylabel('分形维数 (Nf2)')
                ax.set_title('絮团面积与分形维数关系')

            elif chart_type == "纵横比/面积散点图":
                ax.scatter(self.df['Area_um2'], self.df['AspectRatio'], alpha=0.7, c='#795548')
                ax.set_xlabel(f'面积 (μm{SQUARE_SYMBOL})')
                ax.set_ylabel('纵横比')
                ax.set_title('絮团面积与纵横比关系')

            elif chart_type == "纵横比直方图":
                if group_by == "图像":
                    for name, group in self.df.groupby('ImageName'):
                        ax.hist(group['AspectRatio'], bins=bins, alpha=0.5, label=name)
                    ax.legend()
                else:
                    ax.hist(self.df['AspectRatio'], bins=bins, alpha=0.7, color='#607D8B')
                ax.set_xlabel('纵横比')
                ax.set_ylabel('频数')
                ax.set_title('絮团纵横比分布')

            elif chart_type == "凸性直方图":
                if group_by == "图像":
                    for name, group in self.df.groupby('ImageName'):
                        ax.hist(group['Convexity'], bins=bins, alpha=0.5, label=name)
                    ax.legend()
                else:
                    ax.hist(self.df['Convexity'], bins=bins, alpha=0.7, color='#00BCD4')
                ax.set_xlabel('凸性')
                ax.set_ylabel('频数')
                ax.set_title('絮团凸性分布')

            elif chart_type == "紧凑度直方图":
                if group_by == "图像":
                    for name, group in self.df.groupby('ImageName'):
                        ax.hist(group['Compactness'], bins=bins, alpha=0.5, label=name)
                    ax.legend()
                else:
                    ax.hist(self.df['Compactness'], bins=bins, alpha=0.7, color='#8BC34A')
                ax.set_xlabel('紧凑度')
                ax.set_ylabel('频数')
                ax.set_title('絮团紧凑度分布')

            elif chart_type == "粗糙度直方图":
                if group_by == "图像":
                    for name, group in self.df.groupby('ImageName'):
                        ax.hist(group['Roughness'], bins=bins, alpha=0.5, label=name)
                    ax.legend()
                else:
                    ax.hist(self.df['Roughness'], bins=bins, alpha=0.7, color='#FF9800')
                ax.set_xlabel('粗糙度')
                ax.set_ylabel('频数')
                ax.set_title('絮团粗糙度分布')

            elif chart_type == "面积箱线图":
                # 根据选择的分组进行箱线图绘制
                group_option = self.boxplot_combo.currentText()

                if group_option == "按图像":
                    self.df.boxplot(column='Area_um2', by='ImageName', ax=ax)
                    ax.set_title('按图像分组的絮团面积箱线图')

                elif group_option == "按面积范围":
                    # 创建面积范围分组
                    self.df['面积范围'] = pd.cut(self.df['Area_um2'], bins=5,
                                                 labels=['很小', '小', '中等', '大', '很大'])
                    self.df.boxplot(column='AspectRatio', by='面积范围', ax=ax)
                    ax.set_title('不同面积范围的絮团纵横比箱线图')

                elif group_option == "按圆度范围":
                    # 创建圆度范围分组
                    self.df['圆度范围'] = pd.cut(self.df['Circularity'], bins=5,
                                                 labels=['很不圆', '不太圆', '中等', '较圆', '非常圆'])
                    self.df.boxplot(column='Area_um2', by='圆度范围', ax=ax)
                    ax.set_title('不同圆度范围的絮团面积箱线图')

                elif group_option == "按直径范围":
                    # 创建直径范围分组
                    self.df['直径范围'] = pd.cut(self.df['EquivDiameter_um'], bins=5)
                    self.df.boxplot(column='AspectRatio', by='直径范围', ax=ax)
                    ax.set_title('不同直径范围的絮团纵横比箱线图')

                ax.set_xlabel('')
                ax.set_ylabel(f'面积 (μm{SQUARE_SYMBOL})')

            elif chart_type == "参数相关性热图":
                # 选择数值列进行相关性分析
                numeric_cols = ['Area_um2', 'Perimeter_um', 'EquivDiameter_um', 'Circularity',
                                'MaxFeretDiameter_um', 'MinFeretDiameter_um', 'AspectRatio',
                                'Convexity', 'Compactness', 'Roughness', 'Nf2', 'Nf3']

                # 使用更短的列名以适应显示
                short_names = {
                    'Area_um2': f'面积(μm{SQUARE_SYMBOL})',
                    'Perimeter_um': '周长(μm)',
                    'EquivDiameter_um': '等效直径',
                    'Circularity': '圆度',
                    'MaxFeretDiameter_um': '最大Feret径',
                    'MinFeretDiameter_um': '最小Feret径',
                    'AspectRatio': '纵横比',
                    'Convexity': '凸性',
                    'Compactness': '紧凑度',
                    'Roughness': '粗糙度',
                    'Nf2': 'Nf2',
                    'Nf3': 'Nf3'
                }

                # 计算相关性
                corr_df = self.df[numeric_cols].corr()

                # 创建单独的图形用于热图，调整尺寸以适应标签
                self.main_figure.clear()

                # 使用GridSpec来更好地控制布局
                from matplotlib.gridspec import GridSpec
                gs = GridSpec(1, 2, width_ratios=[20, 1], figure=self.main_figure)

                ax = self.main_figure.add_subplot(gs[0])
                cax = self.main_figure.add_subplot(gs[1])

                # 绘制热图
                im = ax.imshow(corr_df.values, cmap='coolwarm', vmin=-1, vmax=1)

                # 设置坐标轴标签
                ax.set_xticks(np.arange(len(numeric_cols)))
                ax.set_yticks(np.arange(len(numeric_cols)))

                # 使用缩写列名
                short_col_names = [short_names[col] for col in numeric_cols]
                ax.set_xticklabels(short_col_names, rotation=45, ha='right', rotation_mode='anchor')
                ax.set_yticklabels(short_col_names)

                # 添加颜色条
                self.main_figure.colorbar(im, cax=cax)

                # 在热图中显示相关系数值
                for i in range(len(numeric_cols)):
                    for j in range(len(numeric_cols)):
                        # 只显示相关系数绝对值大于0.5的
                        val = corr_df.iloc[i, j]
                        if abs(val) > 0.5 and i != j:  # 不显示自相关(对角线)
                            text_color = 'white' if abs(val) > 0.7 else 'black'
                            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)
                        # 对角线显示参数名
                        elif i == j:
                            ax.text(j, i, "1.0", ha="center", va="center", color="white", fontsize=8)

                ax.set_title('参数相关性热图', pad=10)

                # 确保布局适合
                self.main_figure.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.9, wspace=0.05)
                self.main_canvas.draw()
                return  # 直接返回，不使用tight_layout

            # 调整布局
            self.main_figure.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
            self.main_canvas.draw()

        except Exception as e:
            import traceback
            print(f"绘制图表出错: {e}")
            print(traceback.format_exc())

    def export_chart(self):
        """导出当前显示的图表"""
        if self.df is None or self.df.empty:
            QMessageBox.warning(None, "导出错误", "没有数据可以导出")
            return

        try:
            # 选择保存路径
            file_path, _ = QFileDialog.getSaveFileName(
                None, "保存图表", "", "PNG图像 (*.png);;PDF文档 (*.pdf);;SVG图像 (*.svg)")

            if not file_path:
                return

            # 根据文件类型保存
            self.main_figure.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(None, "导出成功", f"图表已成功导出到:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(None, "导出错误", f"导出图表时出错:\n{str(e)}")


class MainWindow(QMainWindow):
    """絮团形态分析系统主窗口"""

    def __init__(self):
        super(MainWindow, self).__init__()

        # 窗口设置
        self.setWindowTitle("絮团形态分析系统")
        self.setGeometry(100, 100, 1400, 900)  # 增加窗口尺寸

        # 中心窗口部件
        self.central_widget = QTabWidget()
        self.setCentralWidget(self.central_widget)

        # 创建各个标签页
        self.setup_processing_tab()
        self.setup_visualization_tab()
        self.setup_statistics_tab()
        self.setup_report_generation_tab()  # 报告生成页面
        self.setup_batch_processing_tab()  # 批量处理页面

        # 版权信息
        copyright_label = QLabel(
            "The software was developed by Ya Wu, with training data jointly annotated by Ya Wu and Ying Chen.")
        copyright_label.setStyleSheet("color: #D81B60; font-style: italic;")
        copyright_label.setAlignment(Qt.AlignCenter)

        # 状态栏
        self.statusBar().addPermanentWidget(copyright_label)
        self.statusBar().showMessage("就绪")

        # 应用样式表
        self.setStyleSheet(DARK_STYLE)

        # 创建勾选图标
        self.create_checkmark_icon()

        # 设置F1快捷键
        self.setup_help_shortcut()

        # 初始化变量
        self.current_df = None
        self.stats = None
        self.process_thread = None
        self.current_pixmap = None
        self.batch_jobs = []  # 存储批量处理任务
        self.current_batch_job_index = -1  # 当前处理的批量任务索引

    def create_checkmark_icon(self):
        """创建勾选图标用于复选框"""
        # 创建一个16x16的图像用于勾选图标
        icon_size = 16
        checkmark = QPixmap(icon_size, icon_size)
        checkmark.fill(Qt.transparent)

        painter = QPainter(checkmark)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.white, 2))

        # 画勾
        points = [QPoint(4, 8), QPoint(7, 12), QPoint(12, 4)]
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

        painter.end()

        # 保存为文件
        checkmark.save("check.png")

    def setup_processing_tab(self):
        """设置处理标签页"""
        processing_tab = QWidget()
        layout = QVBoxLayout(processing_tab)

        # 分割为上方设置区域和下方日志区域
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        # 上方设置区域
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        splitter.addWidget(settings_widget)

        # 文件和路径设置
        path_group = QGroupBox("文件和路径设置")
        path_layout = QGridLayout(path_group)
        settings_layout.addWidget(path_group)

        # 模型路径
        path_layout.addWidget(QLabel("模型路径:"), 0, 0)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setText(
            r"E:\yolov8\yolov8\particle_segmentation\yolov8n-seg-particle-optimized\weights\best_test.pt")
        path_layout.addWidget(self.model_path_edit, 0, 1)
        model_browse_btn = QPushButton("浏览...")
        model_browse_btn.clicked.connect(self.browse_model)
        path_layout.addWidget(model_browse_btn, 0, 2)

        # 图像文件夹
        path_layout.addWidget(QLabel("图像文件夹:"), 1, 0)
        self.image_folder_edit = QLineEdit()
        self.image_folder_edit.setText(r"F:\floc_pretwo\already_1")
        path_layout.addWidget(self.image_folder_edit, 1, 1)
        image_browse_btn = QPushButton("浏览...")
        image_browse_btn.clicked.connect(self.browse_image_folder)
        path_layout.addWidget(image_browse_btn, 1, 2)

        # 输出CSV文件
        path_layout.addWidget(QLabel("输出CSV文件:"), 2, 0)
        self.output_csv_edit = QLineEdit()
        self.output_csv_edit.setText(r"F:\flocresult_test\floc_properties.csv")
        path_layout.addWidget(self.output_csv_edit, 2, 1)
        csv_browse_btn = QPushButton("浏览...")
        csv_browse_btn.clicked.connect(self.browse_output_csv)
        path_layout.addWidget(csv_browse_btn, 2, 2)

        # 输出图像文件夹
        path_layout.addWidget(QLabel("输出图像文件夹:"), 3, 0)
        self.output_image_folder_edit = QLineEdit()
        self.output_image_folder_edit.setText(r"F:\flocresult_test\processed_images")
        path_layout.addWidget(self.output_image_folder_edit, 3, 1)
        output_folder_browse_btn = QPushButton("浏览...")
        output_folder_browse_btn.clicked.connect(self.browse_output_folder)
        path_layout.addWidget(output_folder_browse_btn, 3, 2)

        # 参数设置
        param_group = QGroupBox("参数设置")
        param_layout = QGridLayout(param_group)
        settings_layout.addWidget(param_group)

        # 像素尺寸
        param_layout.addWidget(QLabel("像素尺寸 (微米):"), 0, 0)
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.1, 100.0)
        self.pixel_size_spin.setValue(1.8)
        self.pixel_size_spin.setSingleStep(0.1)
        self.pixel_size_spin.setDecimals(2)
        param_layout.addWidget(self.pixel_size_spin, 0, 1)

        # 置信度阈值
        param_layout.addWidget(QLabel("置信度阈值:"), 1, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        param_layout.addWidget(self.confidence_slider, 1, 1)
        self.confidence_value_label = QLabel("0.50")
        param_layout.addWidget(self.confidence_value_label, 1, 2)
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)

        # 添加过滤阈值设置
        filter_group = QGroupBox("过滤阈值设置")
        filter_layout = QGridLayout(filter_group)
        settings_layout.addWidget(filter_group)

        # 最小面积阈值（微米）
        filter_layout.addWidget(QLabel("最小面积阈值 (微米²):"), 0, 0)
        self.min_area_spin = QDoubleSpinBox()
        self.min_area_spin.setRange(0.1, 10000.0)
        self.min_area_spin.setValue(3.0)  # 默认值设置为10微米²
        self.min_area_spin.setSingleStep(1.0)
        self.min_area_spin.setDecimals(1)
        filter_layout.addWidget(self.min_area_spin, 0, 1)

        # 最小周长阈值（微米）
        filter_layout.addWidget(QLabel("最小周长阈值 (微米):"), 1, 0)
        self.min_perimeter_spin = QDoubleSpinBox()
        self.min_perimeter_spin.setRange(0.1, 5000.0)
        self.min_perimeter_spin.setValue(3.0)  # 默认值设置为20微米
        self.min_perimeter_spin.setSingleStep(1.0)
        self.min_perimeter_spin.setDecimals(1)
        filter_layout.addWidget(self.min_perimeter_spin, 1, 1)

        # 保留这些变量但设置默认值，以便代码其他部分仍然可以访问
        self.min_circularity_spin = QDoubleSpinBox()
        self.min_circularity_spin.setValue(0.0)
        self.min_circularity_spin.setVisible(False)

        self.max_aspect_ratio_spin = QDoubleSpinBox()
        self.max_aspect_ratio_spin.setValue(20.0)
        self.max_aspect_ratio_spin.setVisible(False)

        self.edge_detection_combo = QComboBox()
        self.edge_detection_combo.addItems(["严格 (任何点在边缘)", "宽松 (中心点在边缘)"])
        self.edge_detection_combo.setVisible(False)

        # 是否保存处理图像
        param_layout.addWidget(QLabel("保存处理图像:"), 2, 0)
        self.save_images_check = QCheckBox("√")
        self.save_images_check.setChecked(True)
        param_layout.addWidget(self.save_images_check, 2, 1)

        # 控制按钮区域
        control_layout = QHBoxLayout()
        settings_layout.addLayout(control_layout)

        # 开始处理按钮
        self.start_btn = QPushButton("开始处理")
        self.start_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.start_btn.clicked.connect(self.start_processing)
        control_layout.addWidget(self.start_btn)

        # 停止处理按钮
        self.stop_btn = QPushButton("停止处理")
        self.stop_btn.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_processing)
        control_layout.addWidget(self.stop_btn)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        settings_layout.addWidget(self.progress_bar)

        # 下方日志区域
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        splitter.addWidget(log_widget)

        log_layout.addWidget(QLabel("处理日志:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        # 添加使用教程按钮
        help_btn_layout = QHBoxLayout()
        help_btn = QPushButton("使用教程")
        help_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF80AB;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #FF4081;
            }
        """)
        help_btn.clicked.connect(self.show_help)
        help_btn_layout.addStretch()
        help_btn_layout.addWidget(help_btn)
        help_btn_layout.addStretch()
        log_layout.addLayout(help_btn_layout)

        # 设置分割比例
        splitter.setSizes([400, 400])

        # 将标签页添加到中心窗口
        self.central_widget.addTab(processing_tab, "处理")

    def setup_visualization_tab(self):
        """设置可视化标签页"""
        visualization_tab = QWidget()
        layout = QVBoxLayout(visualization_tab)

        # 分割为上方图像区域和下方控制区域
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        # 上方图像显示区域
        image_scroll = QScrollArea()
        image_scroll.setWidgetResizable(True)
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("处理后的图像将显示在这里")
        image_layout.addWidget(self.image_label)

        image_scroll.setWidget(image_widget)
        splitter.addWidget(image_scroll)

        # 下方控制区域
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        splitter.addWidget(control_widget)

        # 图像浏览控制
        browse_group = QGroupBox("图像浏览")
        browse_layout = QHBoxLayout(browse_group)
        control_layout.addWidget(browse_group)

        self.prev_image_btn = QPushButton("上一张")
        self.prev_image_btn.setEnabled(False)
        self.prev_image_btn.clicked.connect(self.show_prev_image)
        browse_layout.addWidget(self.prev_image_btn)

        self.image_selector = QComboBox()
        self.image_selector.setEnabled(False)
        self.image_selector.currentIndexChanged.connect(self.image_selected)
        browse_layout.addWidget(self.image_selector)

        self.next_image_btn = QPushButton("下一张")
        self.next_image_btn.setEnabled(False)
        self.next_image_btn.clicked.connect(self.show_next_image)
        browse_layout.addWidget(self.next_image_btn)

        # 图像设置
        display_group = QGroupBox("显示设置")
        display_layout = QGridLayout(display_group)
        control_layout.addWidget(display_group)

        # 显示轮廓选项
        display_layout.addWidget(QLabel("显示轮廓:"), 0, 0)
        self.show_contours_check = QCheckBox("√")
        self.show_contours_check.setChecked(True)
        self.show_contours_check.stateChanged.connect(self.update_display_options)
        display_layout.addWidget(self.show_contours_check, 0, 1)

        # 显示ID选项
        display_layout.addWidget(QLabel("显示絮团ID:"), 0, 2)
        self.show_ids_check = QCheckBox("√")
        self.show_ids_check.setChecked(True)
        self.show_ids_check.stateChanged.connect(self.update_display_options)
        display_layout.addWidget(self.show_ids_check, 0, 3)

        # 显示置信度选项
        display_layout.addWidget(QLabel("显示置信度:"), 1, 0)
        self.show_confidence_check = QCheckBox("√")
        self.show_confidence_check.setChecked(True)
        self.show_confidence_check.stateChanged.connect(self.update_display_options)
        display_layout.addWidget(self.show_confidence_check, 1, 1)

        # 放大倍数
        display_layout.addWidget(QLabel("缩放比例:"), 1, 2)
        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(0.1, 5.0)
        self.zoom_spin.setValue(1.0)
        self.zoom_spin.setSingleStep(0.1)
        self.zoom_spin.valueChanged.connect(self.update_zoom)
        display_layout.addWidget(self.zoom_spin, 1, 3)

        # 导出当前图像按钮
        export_layout = QHBoxLayout()
        control_layout.addLayout(export_layout)

        self.export_image_btn = QPushButton("导出当前图像")
        self.export_image_btn.clicked.connect(self.export_current_image)
        self.export_image_btn.setEnabled(False)
        export_layout.addWidget(self.export_image_btn)

        # 设置分割比例
        splitter.setSizes([600, 200])

        # 将标签页添加到中心窗口
        self.central_widget.addTab(visualization_tab, "可视化")

    def setup_statistics_tab(self):
        """设置统计标签页"""
        statistics_tab = QWidget()
        layout = QVBoxLayout(statistics_tab)

        # 创建统计部件
        self.statistics_widget = StatisticsWidget()
        layout.addWidget(self.statistics_widget)

        # 将标签页添加到中心窗口
        self.central_widget.addTab(statistics_tab, "统计")

    def update_confidence_label(self):
        """更新置信度标签的值"""
        value = self.confidence_slider.value() / 100.0
        self.confidence_value_label.setText(f"{value:.2f}")

    def browse_model(self):
        """浏览模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pt *.pth);;所有文件 (*)")
        if file_path:
            self.model_path_edit.setText(file_path)

    def browse_image_folder(self):
        """浏览图像文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择图像文件夹", "")
        if folder_path:
            self.image_folder_edit.setText(folder_path)

    def browse_output_csv(self):
        """浏览输出CSV文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "选择输出CSV文件", "", "CSV文件 (*.csv);;所有文件 (*)")
        if file_path:
            self.output_csv_edit.setText(file_path)

    def browse_output_folder(self):
        """浏览输出图像文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择输出图像文件夹", "")
        if folder_path:
            self.output_image_folder_edit.setText(folder_path)

    def log_message(self, message):
        """添加日志消息"""
        self.log_text.append(message)
        # 滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, progress, message):
        """更新进度条"""
        self.progress_bar.setValue(progress)
        self.statusBar().showMessage(message)

    def update_image(self, pixmap):
        """更新显示的图像"""
        if not pixmap.isNull():
            self.current_pixmap = pixmap

            # 应用缩放
            zoom = self.zoom_spin.value()
            if zoom != 1.0:
                width = int(pixmap.width() * zoom)
                height = int(pixmap.height() * zoom)
                pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.image_label.setPixmap(pixmap)
            self.export_image_btn.setEnabled(True)

    def update_zoom(self):
        """更新图像缩放"""
        if self.current_pixmap and not self.current_pixmap.isNull():
            # 重新应用缩放
            zoom = self.zoom_spin.value()
            width = int(self.current_pixmap.width() * zoom)
            height = int(self.current_pixmap.height() * zoom)
            scaled_pixmap = self.current_pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)

    def export_current_image(self):
        """导出当前显示的图像"""
        if self.current_pixmap and not self.current_pixmap.isNull():
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图像", "", "PNG图像 (*.png);;JPEG图像 (*.jpg);;所有文件 (*)")
            if file_path:
                try:
                    self.current_pixmap.save(file_path)
                    QMessageBox.information(self, "导出成功", f"图像已保存到: {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "导出失败", f"保存图像时出错: {str(e)}")

    def update_stats(self, stats):
        """更新统计信息"""
        self.stats = stats

    def process_done(self, df):
        """处理完成"""
        # 更新UI状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.statusBar().showMessage("处理完成")

        # 保存结果
        self.current_df = df

        # 更新统计信息
        if not df.empty:
            self.statistics_widget.update_statistics(df, self.stats)

            # 切换到统计标签页
            self.central_widget.setCurrentIndex(2)

            # 更新图像浏览控制
            self.update_image_browser()

    def update_image_browser(self):
        """更新图像浏览器列表"""
        self.image_selector.clear()

        if self.current_df is not None and not self.current_df.empty:
            # 获取所有唯一的图像名称
            image_names = self.current_df['ImageName'].unique()

            for name in image_names:
                self.image_selector.addItem(name)

            # 启用浏览控件
            self.image_selector.setEnabled(True)
            self.prev_image_btn.setEnabled(True)
            self.next_image_btn.setEnabled(True)

            # 显示第一张图像
            if self.image_selector.count() > 0:
                self.image_selector.setCurrentIndex(0)

    def image_selected(self, index):
        """当选择图像时调用"""
        if index >= 0 and self.output_image_folder_edit.text():
            image_name = self.image_selector.currentText()
            image_path = os.path.join(self.output_image_folder_edit.text(), f"processed_{image_name}")

            if os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                self.update_image(pixmap)

                # 切换到可视化标签页
                self.central_widget.setCurrentIndex(1)

    def show_prev_image(self):
        """显示上一张图像"""
        current_idx = self.image_selector.currentIndex()
        if current_idx > 0:
            self.image_selector.setCurrentIndex(current_idx - 1)

    def show_next_image(self):
        """显示下一张图像"""
        current_idx = self.image_selector.currentIndex()
        if current_idx < self.image_selector.count() - 1:
            self.image_selector.setCurrentIndex(current_idx + 1)

    def start_processing(self):
        """开始处理"""
        # 检查输入
        model_path = self.model_path_edit.text().strip()
        image_folder = self.image_folder_edit.text().strip()
        output_csv = self.output_csv_edit.text().strip()
        output_image_folder = self.output_image_folder_edit.text().strip()
        save_images = self.save_images_check.isChecked()

        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "输入错误", "请选择有效的模型文件")
            return

        if not image_folder or not os.path.exists(image_folder):
            QMessageBox.warning(self, "输入错误", "请选择有效的图像文件夹")
            return

        if not output_csv:
            QMessageBox.warning(self, "输入错误", "请设置输出CSV文件路径")
            return

        # 如果选择不保存图像，提示用户确认
        if not save_images:
            reply = QMessageBox.question(
                self, '确认设置',
                '您选择了不保存处理后的图像。这将导致无法在"可视化"标签页查看处理结果。确定要继续吗？',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.No:
                return

        # 如果保存图像但未设置输出文件夹，警告用户
        if save_images and not output_image_folder:
            QMessageBox.warning(self, "输入错误", "您选择了保存处理图像，但未设置输出图像文件夹")
            return

        # 获取参数
        pixel_size = self.pixel_size_spin.value()
        confidence = self.confidence_slider.value() / 100.0
        min_area = self.min_area_spin.value()
        min_perimeter = self.min_perimeter_spin.value()
        min_circularity = self.min_circularity_spin.value()
        max_aspect_ratio = self.max_aspect_ratio_spin.value()
        edge_detection_mode = "strict" if self.edge_detection_combo.currentIndex() == 0 else "relaxed"

        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.export_image_btn.setEnabled(False)

        # 创建处理线程
        self.process_thread = ProcessThread(
            model_path, image_folder, output_csv, output_image_folder,
            pixel_size, confidence, save_images, min_area, min_perimeter,
            min_circularity, max_aspect_ratio, edge_detection_mode)

        # 连接信号
        self.process_thread.progress_signal.connect(self.update_progress)
        self.process_thread.log_signal.connect(self.log_message)
        self.process_thread.done_signal.connect(self.process_done)
        self.process_thread.image_signal.connect(self.update_image)
        self.process_thread.stats_signal.connect(self.update_stats)

        # 启动线程
        self.process_thread.start()

        # 更新状态
        self.statusBar().showMessage("正在处理...")
        self.log_message("开始处理絮团图像，请稍候...")

    def stop_processing(self):
        """停止处理"""
        if self.process_thread and self.process_thread.isRunning():
            reply = QMessageBox.question(
                self, '确认停止', '确定要停止当前处理过程吗？\n已处理的数据将被保存。',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                self.log_message("正在停止处理...")
                self.process_thread.stop()
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.statusBar().showMessage("处理已取消")

    def update_display_options(self):
        """更新图像显示选项"""
        if self.current_pixmap and not self.current_pixmap.isNull():
            current_idx = self.image_selector.currentIndex()
            if current_idx >= 0:
                # 重新加载当前图像
                image_name = self.image_selector.currentText()
                image_path = os.path.join(self.output_image_folder_edit.text(), f"processed_{image_name}")

                if os.path.exists(image_path):
                    # 这里我们需要重新绘制图像，但由于我们无法直接修改图像处理代码
                    # 所以暂时只能通过消息提示用户
                    QMessageBox.information(self, "功能提示",
                                            "显示选项已更新。由于技术限制，需要重新处理图像才能应用这些更改。")

                    # 重新应用缩放以确保视觉反馈
                    self.update_zoom()

    def setup_report_generation_tab(self):
        """设置报告生成标签页，用于生成PDF报告"""
        report_tab = QWidget()
        layout = QVBoxLayout(report_tab)

        # 上部设置区域
        settings_group = QGroupBox("报告设置")
        settings_layout = QGridLayout(settings_group)
        layout.addWidget(settings_group)

        # 报告标题
        settings_layout.addWidget(QLabel("报告标题:"), 0, 0)
        self.report_title_edit = QLineEdit("絮团形态分析报告")
        settings_layout.addWidget(self.report_title_edit, 0, 1)

        # 报告作者
        settings_layout.addWidget(QLabel("报告作者:"), 1, 0)
        self.report_author_edit = QLineEdit()
        settings_layout.addWidget(self.report_author_edit, 1, 1)

        # 实验描述
        settings_layout.addWidget(QLabel("实验描述:"), 2, 0)
        self.report_desc_edit = QTextEdit()
        self.report_desc_edit.setMaximumHeight(100)
        settings_layout.addWidget(self.report_desc_edit, 2, 1)

        # 报告输出路径
        settings_layout.addWidget(QLabel("输出路径:"), 3, 0)
        path_layout = QHBoxLayout()
        self.report_path_edit = QLineEdit()
        self.report_path_edit.setText(os.path.expanduser("~/Desktop/floc_report.pdf"))
        path_layout.addWidget(self.report_path_edit)

        report_browse_btn = QPushButton("浏览...")
        report_browse_btn.clicked.connect(self.browse_report_path)
        path_layout.addWidget(report_browse_btn)
        settings_layout.addLayout(path_layout, 3, 1)

        # 中部内容选择区域
        content_group = QGroupBox("报告内容")
        content_layout = QGridLayout(content_group)
        layout.addWidget(content_group)

        # 选择要包含的图表类型
        content_layout.addWidget(QLabel("包含图表:"), 0, 0)
        self.chart_types_list = QListWidget()
        self.chart_types_list.setSelectionMode(QAbstractItemView.MultiSelection)
        chart_items = [
            "面积分布直方图", "直径分布直方图", "圆度分布直方图",
            "面积/直径散点图", "圆度/直径散点图", "Feret长短径散点图",
            "参数相关性热图", "面积箱线图", "处理结果分布饼图"
        ]
        for item in chart_items:
            list_item = QListWidgetItem(item)
            list_item.setCheckState(Qt.Checked)
            self.chart_types_list.addItem(list_item)
        content_layout.addWidget(self.chart_types_list, 0, 1, 3, 1)

        # 选择要包含的统计数据
        content_layout.addWidget(QLabel("包含统计:"), 3, 0)
        self.stats_types_list = QListWidget()
        self.stats_types_list.setSelectionMode(QAbstractItemView.MultiSelection)
        stats_items = [
            "基本统计量", "尺寸分布", "形状指标", "分形维数分析",
            "各参数间相关性", "图像处理统计", "阈值筛选统计",
            "形状分类统计", "边缘检测统计"
        ]
        for item in stats_items:
            list_item = QListWidgetItem(item)
            list_item.setCheckState(Qt.Checked)
            self.stats_types_list.addItem(list_item)
        content_layout.addWidget(self.stats_types_list, 3, 1, 3, 1)

        # 数据源选择
        source_group = QGroupBox("数据源")
        source_layout = QVBoxLayout(source_group)

        # 使用当前数据或选择文件
        self.use_current_data_radio = QRadioButton("使用当前已加载数据")
        self.use_current_data_radio.setChecked(True)
        source_layout.addWidget(self.use_current_data_radio)

        self.use_file_data_radio = QRadioButton("使用指定数据文件")
        source_layout.addWidget(self.use_file_data_radio)

        file_select_layout = QHBoxLayout()
        self.report_data_file_edit = QLineEdit()
        self.report_data_file_edit.setEnabled(False)
        file_select_layout.addWidget(self.report_data_file_edit)

        self.report_data_browse_btn = QPushButton("浏览...")
        self.report_data_browse_btn.setEnabled(False)
        self.report_data_browse_btn.clicked.connect(self.browse_report_data)
        file_select_layout.addWidget(self.report_data_browse_btn)
        source_layout.addLayout(file_select_layout)

        # 连接单选按钮信号
        self.use_current_data_radio.toggled.connect(self.toggle_report_data_source)
        self.use_file_data_radio.toggled.connect(self.toggle_report_data_source)

        content_layout.addWidget(source_group, 0, 2, 6, 1)

        # 底部控制按钮
        control_layout = QHBoxLayout()
        layout.addLayout(control_layout)

        # 预览报告按钮
        preview_report_btn = QPushButton("预览报告")
        preview_report_btn.clicked.connect(self.preview_report)
        control_layout.addWidget(preview_report_btn)

        # 生成报告按钮
        generate_report_btn = QPushButton("生成报告")
        generate_report_btn.clicked.connect(self.generate_report)
        control_layout.addWidget(generate_report_btn)

        # 将标签页添加到中心窗口
        self.central_widget.addTab(report_tab, "报告生成")

    def setup_batch_processing_tab(self):
        """设置批量处理标签页，用于批量处理多个图像文件夹"""
        batch_tab = QWidget()
        layout = QVBoxLayout(batch_tab)

        # 上部任务区域
        tasks_group = QGroupBox("处理任务")
        tasks_layout = QVBoxLayout(tasks_group)
        layout.addWidget(tasks_group)

        # 任务列表
        self.batch_tasks_list = QListWidget()
        self.batch_tasks_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.batch_tasks_list.setDragDropMode(QAbstractItemView.InternalMove)
        tasks_layout.addWidget(self.batch_tasks_list)

        # 添加/删除任务按钮
        buttons_layout = QHBoxLayout()
        tasks_layout.addLayout(buttons_layout)

        self.add_task_btn = QPushButton("添加任务")
        self.add_task_btn.setIcon(QIcon.fromTheme("list-add"))
        self.add_task_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF80AB;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #FF4081;
            }
            QPushButton:pressed {
                background-color: #F50057;
            }
        """)
        self.add_task_btn.clicked.connect(self.add_batch_task)
        buttons_layout.addWidget(self.add_task_btn)

        self.remove_task_btn = QPushButton("移除任务")
        self.remove_task_btn.setIcon(QIcon.fromTheme("list-remove"))
        self.remove_task_btn.setStyleSheet("""
            QPushButton {
                background-color: #FFCDD2;
                color: #D81B60;
                border: none;
                border-radius: 4px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #F8BBD0;
            }
            QPushButton:pressed {
                background-color: #F48FB1;
            }
        """)
        self.remove_task_btn.clicked.connect(self.remove_batch_task)
        buttons_layout.addWidget(self.remove_task_btn)

        # 批量处理设置
        settings_group = QGroupBox("批量处理设置")
        settings_layout = QFormLayout(settings_group)
        layout.addWidget(settings_group)

        # 模型
        model_layout = QHBoxLayout()
        self.batch_model_path_edit = QLineEdit()
        self.batch_model_path_edit.setPlaceholderText("YOLOv8模型路径")
        self.batch_model_path_edit.setText(
            r"E:\yolov8\yolov8\particle_segmentation\yolov8n-seg-particle-optimized\weights\best_test.pt")
        model_layout.addWidget(self.batch_model_path_edit)

        self.batch_model_browse_btn = QPushButton("浏览...")
        self.batch_model_browse_btn.clicked.connect(self.browse_batch_model)
        model_layout.addWidget(self.batch_model_browse_btn)
        settings_layout.addRow("模型路径:", model_layout)

        # 置信度
        self.batch_confidence_slider = QSlider(Qt.Horizontal)
        self.batch_confidence_slider.setRange(10, 95)
        self.batch_confidence_slider.setValue(50)
        self.batch_confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.batch_confidence_slider.setTickInterval(10)
        self.batch_confidence_label = QLabel("0.50")

        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(self.batch_confidence_slider)
        confidence_layout.addWidget(self.batch_confidence_label)
        settings_layout.addRow("置信度阈值:", confidence_layout)

        # 连接置信度滑块信号
        self.batch_confidence_slider.valueChanged.connect(
            lambda v: self.batch_confidence_label.setText(f"{v / 100:.2f}"))

        # 像素尺寸
        self.batch_pixel_size_edit = QLineEdit("1.8")
        self.batch_pixel_size_edit.setValidator(QDoubleValidator(0.001, 1000.0, 3))
        settings_layout.addRow("像素尺寸 (μm):", self.batch_pixel_size_edit)

        # 保存图像
        self.batch_save_images_check = QCheckBox("保存处理后的图像")
        self.batch_save_images_check.setChecked(True)
        settings_layout.addRow("", self.batch_save_images_check)

        # 控制按钮
        control_layout = QHBoxLayout()
        layout.addLayout(control_layout)

        self.start_batch_btn = QPushButton("开始批量处理")
        self.start_batch_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.start_batch_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF80AB;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF4081;
            }
            QPushButton:pressed {
                background-color: #F50057;
            }
        """)
        self.start_batch_btn.clicked.connect(self.start_batch_processing)
        control_layout.addWidget(self.start_batch_btn)

        self.stop_batch_btn = QPushButton("停止批量处理")
        self.stop_batch_btn.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_batch_btn.setStyleSheet("""
            QPushButton {
                background-color: #EC407A;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E91E63;
            }
            QPushButton:pressed {
                background-color: #D81B60;
            }
        """)
        self.stop_batch_btn.clicked.connect(self.stop_batch_processing)
        self.stop_batch_btn.setEnabled(False)
        control_layout.addWidget(self.stop_batch_btn)

        # 进度和日志区域
        progress_group = QGroupBox("批量处理进度")
        progress_layout = QVBoxLayout(progress_group)
        layout.addWidget(progress_group)

        # 总进度条
        self.batch_progress = QProgressBar()
        self.batch_progress.setTextVisible(True)
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setValue(0)
        progress_layout.addWidget(self.batch_progress)

        # 当前任务进度
        self.current_task_label = QLabel("等待开始...")
        progress_layout.addWidget(self.current_task_label)

        # 批处理日志
        self.batch_log = QTextEdit()
        self.batch_log.setReadOnly(True)
        self.batch_log.setMinimumHeight(150)
        progress_layout.addWidget(self.batch_log)

        # 设置变量
        self.batch_threads = []
        self.current_batch_task = 0
        self.batch_processing = False

        # 将标签页添加到主窗口
        self.central_widget.addTab(batch_tab, "批量处理")

    # 添加报告生成相关的方法
    def browse_report_path(self):
        """浏览报告输出路径"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "选择报告输出位置", "", "PDF文件 (*.pdf);;所有文件 (*)")
        if file_path:
            self.report_path_edit.setText(file_path)

    def browse_report_data(self):
        """浏览报告数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "", "CSV文件 (*.csv);;所有文件 (*)")
        if file_path:
            self.report_data_file_edit.setText(file_path)

    def toggle_report_data_source(self):
        """切换报告数据源"""
        use_file = self.use_file_data_radio.isChecked()
        self.report_data_file_edit.setEnabled(use_file)
        self.report_data_browse_btn.setEnabled(use_file)

    def preview_report(self):
        """预览报告"""
        if self.use_current_data_radio.isChecked() and (self.current_df is None or self.current_df.empty):
            QMessageBox.warning(self, "数据缺失", "当前没有已加载的数据可用于预览报告。请先处理数据或选择数据文件。")
            return

        if self.use_file_data_radio.isChecked() and not os.path.exists(self.report_data_file_edit.text()):
            QMessageBox.warning(self, "文件错误", "指定的数据文件不存在，请选择有效的CSV文件。")
            return

        QMessageBox.information(self, "预览功能",
                                "报告预览功能正在建设中...\n\n报告将包含您所选择的所有图表和统计数据。")

    def generate_report(self):
        """生成PDF报告"""
        # 检查数据是否可用
        if self.use_current_data_radio.isChecked():
            if self.current_df is None or self.current_df.empty:
                QMessageBox.warning(self, "数据缺失", "当前没有已加载的数据可用于生成报告。请先处理数据或选择数据文件。")
                return
            df = self.current_df
        else:
            file_path = self.report_data_file_edit.text()
            if not os.path.exists(file_path):
                QMessageBox.warning(self, "文件错误", "指定的数据文件不存在，请选择有效的CSV文件。")
                return
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                QMessageBox.critical(self, "读取错误", f"读取数据文件时出错:\n{str(e)}")
                return

        # 获取报告设置
        report_title = self.report_title_edit.text()
        report_author = self.report_author_edit.text()
        report_desc = self.report_desc_edit.toPlainText()
        output_path = self.report_path_edit.text()

        # 获取选中的图表和统计类型
        selected_charts = []
        for i in range(self.chart_types_list.count()):
            item = self.chart_types_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_charts.append(item.text())

        selected_stats = []
        for i in range(self.stats_types_list.count()):
            item = self.stats_types_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_stats.append(item.text())

        # 显示正在生成的消息
        self.statusBar().showMessage(f"正在生成报告: {report_title}...")

        try:
            # 创建临时文件夹存储图表
            temp_dir = os.path.join(os.path.dirname(output_path), "temp_report_images")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # 生成图表图像
            chart_images = []

            # 进度对话框
            progress = QProgressDialog("正在生成报告...", "取消", 0, 100, self)
            progress.setWindowTitle("生成报告")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # 生成各种图表的图像文件
            progress.setValue(10)
            QApplication.processEvents()

            # 1. 面积直方图
            if "面积分布直方图" in selected_charts and 'Area_um2' in df.columns:
                plt.figure(figsize=(8, 5))
                plt.hist(df['Area_um2'].dropna(), bins=20, alpha=0.7,
                         color='#FF80AB', edgecolor='#D81B60')
                plt.xlabel('面积 (μm²)')
                plt.ylabel('频数')
                plt.title('絮团面积分布')
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()

                # 添加平均值和中位数线
                mean_val = df['Area_um2'].mean()
                median_val = df['Area_um2'].median()
                plt.axvline(mean_val, color='#D81B60', linestyle='--', linewidth=1.5, label=f'平均值: {mean_val:.2f}')
                plt.axvline(median_val, color='#880E4F', linestyle='-.', linewidth=1.5,
                            label=f'中位数: {median_val:.2f}')
                plt.legend()

                # 保存图像
                area_hist_path = os.path.join(temp_dir, "area_histogram.png")
                plt.savefig(area_hist_path, dpi=150)
                plt.close()
                chart_images.append(("面积分布直方图", area_hist_path))

            progress.setValue(30)
            QApplication.processEvents()

            # 2. 直径直方图
            if "直径分布直方图" in selected_charts and 'EquivDiameter_um' in df.columns:
                plt.figure(figsize=(8, 5))
                plt.hist(df['EquivDiameter_um'].dropna(), bins=20, alpha=0.7,
                         color='#F06292', edgecolor='#D81B60')
                plt.xlabel('等效直径 (μm)')
                plt.ylabel('频数')
                plt.title('絮团直径分布')
                plt.grid(True, linestyle='--', alpha=0.3)

                # 添加平均值和中位数线
                mean_val = df['EquivDiameter_um'].mean()
                median_val = df['EquivDiameter_um'].median()
                plt.axvline(mean_val, color='#D81B60', linestyle='--', linewidth=1.5, label=f'平均值: {mean_val:.2f}')
                plt.axvline(median_val, color='#880E4F', linestyle='-.', linewidth=1.5,
                            label=f'中位数: {median_val:.2f}')
                plt.legend()
                plt.tight_layout()

                # 保存图像
                diam_hist_path = os.path.join(temp_dir, "diameter_histogram.png")
                plt.savefig(diam_hist_path, dpi=150)
                plt.close()
                chart_images.append(("直径分布直方图", diam_hist_path))

            progress.setValue(50)
            QApplication.processEvents()

            # 3. 面积/直径散点图
            if "面积/直径散点图" in selected_charts and 'Area_um2' in df.columns and 'EquivDiameter_um' in df.columns:
                plt.figure(figsize=(8, 5))
                plt.scatter(df['Area_um2'], df['EquivDiameter_um'], alpha=0.7,
                            color='#EC407A', edgecolor='#D81B60')

                # 添加趋势线
                if len(df) > 1:
                    z = np.polyfit(df['Area_um2'], df['EquivDiameter_um'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(df['Area_um2'].min(), df['Area_um2'].max(), 100)
                    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'y = {z[0]:.4f}x + {z[1]:.2f}')

                plt.xlabel('面积 (μm²)')
                plt.ylabel('等效直径 (μm)')
                plt.title('絮团面积与直径关系')
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.legend()
                plt.tight_layout()

                # 保存图像
                scatter_path = os.path.join(temp_dir, "area_diameter_scatter.png")
                plt.savefig(scatter_path, dpi=150)
                plt.close()
                chart_images.append(("面积/直径散点图", scatter_path))

            progress.setValue(60)
            QApplication.processEvents()

            # 4. 圆度/直径散点图
            if "圆度/直径散点图" in selected_charts and 'Circularity' in df.columns and 'EquivDiameter_um' in df.columns:
                plt.figure(figsize=(8, 5))
                plt.scatter(df['EquivDiameter_um'], df['Circularity'], alpha=0.7,
                            color='#E91E63', edgecolor='#C2185B')

                # 添加趋势线
                if len(df) > 1:
                    z = np.polyfit(df['EquivDiameter_um'], df['Circularity'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(df['EquivDiameter_um'].min(), df['EquivDiameter_um'].max(), 100)
                    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'y = {z[0]:.4f}x + {z[1]:.2f}')

                plt.xlabel('等效直径 (μm)')
                plt.ylabel('圆度')
                plt.title('絮团圆度与直径关系')
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.legend()
                plt.tight_layout()

                # 保存图像
                circ_diam_path = os.path.join(temp_dir, "circularity_diameter_scatter.png")
                plt.savefig(circ_diam_path, dpi=150)
                plt.close()
                chart_images.append(("圆度/直径散点图", circ_diam_path))

            progress.setValue(70)
            QApplication.processEvents()

            # 5. Feret长短径散点图
            if "Feret长短径散点图" in selected_charts and 'MaxFeretDiameter_um' in df.columns and 'MinFeretDiameter_um' in df.columns:
                plt.figure(figsize=(8, 5))
                plt.scatter(df['MaxFeretDiameter_um'], df['MinFeretDiameter_um'], alpha=0.7,
                            color='#D81B60', edgecolor='#AD1457')

                # 添加趋势线
                if len(df) > 1:
                    z = np.polyfit(df['MaxFeretDiameter_um'], df['MinFeretDiameter_um'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(df['MaxFeretDiameter_um'].min(), df['MaxFeretDiameter_um'].max(), 100)
                    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'y = {z[0]:.4f}x + {z[1]:.2f}')

                plt.xlabel('最大Feret直径 (μm)')
                plt.ylabel('最小Feret直径 (μm)')
                plt.title('Feret长短径关系')
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.legend()
                plt.tight_layout()

                # 保存图像
                feret_path = os.path.join(temp_dir, "feret_diameters_scatter.png")
                plt.savefig(feret_path, dpi=150)
                plt.close()
                chart_images.append(("Feret长短径散点图", feret_path))

            progress.setValue(80)
            QApplication.processEvents()

            # 6. 参数相关性热图
            if "参数相关性热图" in selected_charts:
                # 选择主要参数进行相关性分析
                corr_params = ['Area_um2', 'Perimeter_um', 'EquivDiameter_um', 'Circularity',
                               'MaxFeretDiameter_um', 'MinFeretDiameter_um', 'AspectRatio',
                               'Convexity', 'Compactness', 'Roughness', 'Nf2', 'Nf3']

                # 过滤出存在的列
                corr_params = [param for param in corr_params if param in df.columns]

                if len(corr_params) > 1:  # 至少需要两个参数才能计算相关性
                    corr_df = df[corr_params].corr()

                    plt.figure(figsize=(10, 8))
                    sns.heatmap(corr_df, annot=True, cmap='RdPu', fmt='.2f', linewidths=0.5,
                                cbar_kws={'label': '相关系数'})
                    plt.title('絮团参数相关性热图')
                    plt.tight_layout()

                    # 保存图像
                    corr_path = os.path.join(temp_dir, "correlation_heatmap.png")
                    plt.savefig(corr_path, dpi=150)
                    plt.close()
                    chart_images.append(("参数相关性热图", corr_path))

            progress.setValue(85)
            QApplication.processEvents()

            # 7. 面积箱线图
            if "面积箱线图" in selected_charts and 'Area_um2' in df.columns and 'ImageName' in df.columns:
                # 按图像分组
                if df['ImageName'].nunique() > 1 and df['ImageName'].nunique() <= 10:  # 限制图像数量，避免图表过于复杂
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x='ImageName', y='Area_um2', data=df, palette='RdPu')
                    plt.xlabel('图像名称')
                    plt.ylabel('面积 (μm²)')
                    plt.title('不同图像中絮团面积分布')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    # 保存图像
                    box_path = os.path.join(temp_dir, "area_boxplot.png")
                    plt.savefig(box_path, dpi=150)
                    plt.close()
                    chart_images.append(("面积箱线图", box_path))

            progress.setValue(90)
            QApplication.processEvents()

            # 8. 处理结果分布饼图
            if "处理结果分布饼图" in selected_charts and 'Circularity' in df.columns:
                # 创建分类
                df['shape_category'] = pd.cut(
                    df['Circularity'],
                    bins=[0, 0.3, 0.6, 0.8, 1.0],
                    labels=['不规则 (<0.3)', '中等规则 (0.3-0.6)', '规则 (0.6-0.8)', '高度规则 (>0.8)']
                )

                # 计算各类别数量
                shape_counts = df['shape_category'].value_counts()

                plt.figure(figsize=(8, 8))
                plt.pie(shape_counts, labels=shape_counts.index, autopct='%1.1f%%',
                        colors=['#FF80AB', '#F06292', '#E91E63', '#C2185B'],
                        startangle=90, shadow=True, explode=[0.05, 0, 0, 0])
                plt.title('絮团形状分布 (基于圆度)')
                plt.axis('equal')  # 确保饼图是圆的
                plt.tight_layout()

                # 保存图像
                pie_path = os.path.join(temp_dir, "shape_distribution_pie.png")
                plt.savefig(pie_path, dpi=150)
                plt.close()
                chart_images.append(("处理结果分布饼图", pie_path))

            progress.setValue(95)
            QApplication.processEvents()

            # 简单的HTML报告
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>{report_title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
                    h1 {{ color: #D81B60; text-align: center; }}
                    h2 {{ color: #D81B60; margin-top: 20px; }}
                    h3 {{ color: #E91E63; }}
                    .author {{ text-align: center; color: #777; font-style: italic; margin-bottom: 30px; }}
                    .description {{ background-color: #FFEBEE; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #FFCDD2; padding: 8px; text-align: left; }}
                    th {{ background-color: #FFCDD2; color: #880E4F; }}
                    tr:nth-child(even) {{ background-color: #FFF5F7; }}
                    .chart {{ text-align: center; margin: 20px 0; }}
                    .chart img {{ max-width: 100%; height: auto; }}
                    .footer {{ text-align: center; margin-top: 50px; font-size: 0.8em; color: #777; }}
                </style>
            </head>
            <body>
                <h1>{report_title}</h1>
                <div class="author">作者: {report_author} | 生成日期: {time.strftime('%Y-%m-%d %H:%M:%S')}</div>
            """

            # 添加实验描述
            if report_desc:
                html_content += f"""
                <div class="description">
                    <h3>实验描述:</h3>
                    <p>{report_desc}</p>
                </div>
                """

            # 添加基本统计量
            if "基本统计量" in selected_stats:
                html_content += """
                <h2>1. 基本统计量</h2>
                <table>
                    <tr>
                        <th>参数</th>
                        <th>样本数</th>
                        <th>均值</th>
                        <th>中位数</th>
                        <th>标准差</th>
                        <th>最小值</th>
                        <th>最大值</th>
                    </tr>
                """

                # 要包含的参数列表
                params = [
                    ('Area_um2', '面积 (μm²)'),
                    ('Perimeter_um', '周长 (μm)'),
                    ('EquivDiameter_um', '等效直径 (μm)'),
                    ('Circularity', '圆度'),
                    ('MaxFeretDiameter_um', '最大Feret直径 (μm)'),
                    ('MinFeretDiameter_um', '最小Feret直径 (μm)'),
                    ('AspectRatio', '纵横比'),
                    ('Convexity', '凸性'),
                    ('Compactness', '紧凑度'),
                    ('Roughness', '粗糙度'),
                    ('Nf2', '二维分形维数'),
                    ('Nf3', '三维分形维数')
                ]

                for param_id, param_name in params:
                    if param_id in df.columns:
                        data = df[param_id].dropna()
                        html_content += f"""
                        <tr>
                            <td>{param_name}</td>
                            <td>{len(data)}</td>
                            <td>{data.mean():.2f}</td>
                            <td>{data.median():.2f}</td>
                            <td>{data.std():.2f}</td>
                            <td>{data.min():.2f}</td>
                            <td>{data.max():.2f}</td>
                        </tr>
                        """

                html_content += "</table>"

            # 添加图表
            if chart_images:
                html_content += "<h2>2. 图表分析</h2>"

                for i, (chart_name, chart_path) in enumerate(chart_images):
                    html_content += f"""
                    <div class="chart">
                        <h3>2.{i + 1} {chart_name}</h3>
                        <img src="{chart_path}" alt="{chart_name}">
                    </div>
                    """

            # 添加尺寸分布分析
            if "尺寸分布" in selected_stats and 'EquivDiameter_um' in df.columns:
                html_content += "<h2>3. 尺寸分布分析</h2>"

                # 创建尺寸分布区间
                bins = [0, 50, 100, 200, 500, 1000, float('inf')]
                labels = ['<50μm', '50-100μm', '100-200μm', '200-500μm', '500-1000μm', '>1000μm']

                df['size_group'] = pd.cut(df['EquivDiameter_um'], bins=bins, labels=labels)
                size_dist = df['size_group'].value_counts().sort_index()

                # 计算百分比
                total = size_dist.sum()
                percentages = size_dist / total * 100

                html_content += """
                <table>
                    <tr>
                        <th>尺寸区间</th>
                        <th>数量</th>
                        <th>百分比</th>
                    </tr>
                """

                for label, count in size_dist.items():
                    html_content += f"""
                    <tr>
                        <td>{label}</td>
                        <td>{count}</td>
                        <td>{percentages[label]:.2f}%</td>
                    </tr>
                    """

                html_content += "</table>"

            # 添加形状分类统计
            if "形状分类统计" in selected_stats and 'Circularity' in df.columns:
                html_content += "<h2>4. 形状分类统计</h2>"

                # 创建分类
                df['shape_category'] = pd.cut(
                    df['Circularity'],
                    bins=[0, 0.3, 0.6, 0.8, 1.0],
                    labels=['不规则 (<0.3)', '中等规则 (0.3-0.6)', '规则 (0.6-0.8)', '高度规则 (>0.8)']
                )

                # 计算各类别数量
                shape_counts = df['shape_category'].value_counts().sort_index()
                shape_percentages = shape_counts / shape_counts.sum() * 100

                html_content += """
                <table>
                    <tr>
                        <th>形状分类</th>
                        <th>数量</th>
                        <th>百分比</th>
                        <th>平均面积 (μm²)</th>
                        <th>平均直径 (μm)</th>
                    </tr>
                """

                for category in shape_counts.index:
                    category_df = df[df['shape_category'] == category]
                    mean_area = category_df['Area_um2'].mean()
                    mean_diameter = category_df['EquivDiameter_um'].mean()

                    html_content += f"""
                    <tr>
                        <td>{category}</td>
                        <td>{shape_counts[category]}</td>
                        <td>{shape_percentages[category]:.2f}%</td>
                        <td>{mean_area:.2f}</td>
                        <td>{mean_diameter:.2f}</td>
                    </tr>
                    """

                html_content += "</table>"

            # 添加阈值筛选统计
            if "阈值筛选统计" in selected_stats:
                html_content += "<h2>5. 阈值筛选统计</h2>"

                # 计算各阈值的统计信息
                html_content += """
                <p>以下是当前处理中使用的阈值参数及其对絮团检测的影响。</p>
                <table>
                    <tr>
                        <th>参数</th>
                        <th>阈值设置</th>
                        <th>影响描述</th>
                    </tr>
                """

                # 添加各阈值参数
                html_content += f"""
                <tr>
                    <td>最小面积阈值</td>
                    <td>{self.min_area_spin.value()} 微米</td>
                    <td>过滤掉面积小于该值的絮团，减少噪声和小碎片的影响</td>
                </tr>
                <tr>
                    <td>最小周长阈值</td>
                    <td>{self.min_perimeter_spin.value()} 微米</td>
                    <td>过滤掉周长过短的絮团，提高轮廓提取的准确性</td>
                </tr>
                <tr>
                    <td>最小圆度阈值</td>
                    <td>{self.min_circularity_spin.value():.2f}</td>
                    <td>过滤掉圆度过低的絮团，可以去除形状不规则的对象</td>
                </tr>
                <tr>
                    <td>最大纵横比阈值</td>
                    <td>{self.max_aspect_ratio_spin.value():.1f}</td>
                    <td>过滤掉纵横比过大的絮团，去除线状或长条形对象</td>
                </tr>
                <tr>
                    <td>边缘检测模式</td>
                    <td>{"严格" if self.edge_detection_combo.currentIndex() == 0 else "宽松"}</td>
                    <td>{"任何点在图像边缘则过滤" if self.edge_detection_combo.currentIndex() == 0 else "仅当中心点在边缘时过滤"}</td>
                </tr>
                <tr>
                    <td>置信度阈值</td>
                    <td>{self.confidence_slider.value() / 100.0:.2f}</td>
                    <td>过滤掉置信度低于该值的检测结果，提高检测的可靠性</td>
                </tr>
                </table>
                """

            # 添加边缘检测统计
            if "边缘检测统计" in selected_stats and 'ImageName' in df.columns:
                html_content += "<h2>6. 边缘检测统计</h2>"

                # 计算每张图像的絮团数量
                image_counts = df['ImageName'].value_counts().sort_index()

                html_content += """
                <p>以下是各图像中有效絮团的统计信息。</p>
                <table>
                    <tr>
                        <th>图像名称</th>
                        <th>有效絮团数量</th>
                        <th>平均面积 (μm²)</th>
                        <th>平均直径 (μm)</th>
                        <th>平均圆度</th>
                    </tr>
                """

                for image_name in image_counts.index:
                    image_df = df[df['ImageName'] == image_name]
                    mean_area = image_df['Area_um2'].mean()
                    mean_diameter = image_df['EquivDiameter_um'].mean()
                    mean_circularity = image_df['Circularity'].mean()

                    html_content += f"""
                    <tr>
                        <td>{image_name}</td>
                        <td>{image_counts[image_name]}</td>
                        <td>{mean_area:.2f}</td>
                        <td>{mean_diameter:.2f}</td>
                        <td>{mean_circularity:.2f}</td>
                    </tr>
                    """

                html_content += "</table>"

            # 添加页脚
            html_content += """
                <div class="footer">
                    生成工具: 絮团形态分析系统<br>
                    The software was developed by Ya Wu, with training data jointly annotated by Ya Wu and Ying Chen.
                </div>
            </body>
            </html>
            """

            # 将HTML保存为文件
            html_path = os.path.splitext(output_path)[0] + ".html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            progress.setValue(90)
            QApplication.processEvents()

            # 将HTML转换为PDF (这里模拟,实际需要安装wkhtmltopdf或其他工具)
            # 此处用示例代码替代实际的HTML到PDF转换，仅显示消息
            # 实际项目中可以使用weasyprint、wkhtmltopdf等工具进行转换

            # 这里只创建一个包含相同信息的简单示例PDF
            try:
                # 尝试将html内容转换为pdf
                QMessageBox.information(self, "报告已生成",
                                        f"HTML报告已生成: {html_path}\n\n注意: 如果不想要某个指标记得取消勾选，例如每张图片的有效絮团统计情况！！")
            except:
                # 如果失败则只提供html版本
                QMessageBox.information(self, "报告已生成",
                                        f"HTML报告已生成: {html_path}\n\n如需PDF版本，请将HTML手动转换为PDF。")

            # 尝试打开生成的HTML文件
            if os.path.exists(html_path):
                if sys.platform == 'win32':
                    os.startfile(html_path)
                elif sys.platform == 'darwin':  # macOS
                    os.system(f"open {html_path}")
                else:  # Linux
                    os.system(f"xdg-open {html_path}")

            progress.setValue(100)

        except Exception as e:
            QMessageBox.critical(self, "生成错误", f"生成报告时出错:\n{str(e)}")
            import traceback
            print(traceback.format_exc())
        finally:
            self.statusBar().showMessage("就绪")

    # 添加批量处理相关的方法
    def browse_batch_model(self):
        """浏览批量处理模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pt *.pth);;所有文件 (*)")
        if file_path:
            self.batch_model_path_edit.setText(file_path)

    def add_batch_task(self):
        """添加批量处理任务"""
        # 创建对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("添加批量处理任务")
        dialog.setMinimumWidth(500)

        layout = QVBoxLayout(dialog)

        # 输入路径
        form_layout = QFormLayout()
        layout.addLayout(form_layout)

        # 任务名称
        task_name_edit = QLineEdit()
        task_name_edit.setPlaceholderText("任务名称（可选）")
        form_layout.addRow("任务名称:", task_name_edit)

        # 输入文件夹
        input_layout = QHBoxLayout()
        input_folder_edit = QLineEdit()
        input_folder_edit.setPlaceholderText("输入图像文件夹路径")
        input_layout.addWidget(input_folder_edit)

        browse_input_btn = QPushButton("浏览...")
        browse_input_btn.clicked.connect(lambda: self._browse_folder(input_folder_edit, "选择输入图像文件夹"))
        input_layout.addWidget(browse_input_btn)
        form_layout.addRow("输入文件夹:", input_layout)

        # 输出CSV
        output_csv_layout = QHBoxLayout()
        output_csv_edit = QLineEdit()
        output_csv_edit.setPlaceholderText("输出CSV文件路径")
        output_csv_layout.addWidget(output_csv_edit)

        browse_csv_btn = QPushButton("浏览...")
        browse_csv_btn.clicked.connect(
            lambda: self._browse_save_file(output_csv_edit, "选择输出CSV文件", "CSV文件 (*.csv)"))
        output_csv_layout.addWidget(browse_csv_btn)
        form_layout.addRow("输出CSV:", output_csv_layout)

        # 输出图像文件夹
        output_image_layout = QHBoxLayout()
        output_image_edit = QLineEdit()
        output_image_edit.setPlaceholderText("输出图像文件夹路径")
        output_image_layout.addWidget(output_image_edit)

        browse_output_btn = QPushButton("浏览...")
        browse_output_btn.clicked.connect(lambda: self._browse_folder(output_image_edit, "选择输出图像文件夹"))
        output_image_layout.addWidget(browse_output_btn)
        form_layout.addRow("输出图像:", output_image_layout)

        # 按钮
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("添加")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(ok_btn)

        # 处理结果
        if dialog.exec_() == QDialog.Accepted:
            input_folder = input_folder_edit.text()
            output_csv = output_csv_edit.text()
            output_image = output_image_edit.text()
            task_name = task_name_edit.text()

            # 验证路径
            if not input_folder:
                QMessageBox.warning(self, "输入错误", "请指定输入图像文件夹路径！")
                return
            if not os.path.isdir(input_folder):
                QMessageBox.warning(self, "输入错误", "指定的输入文件夹不存在！")
                return

            if not output_csv:
                QMessageBox.warning(self, "输入错误", "请指定输出CSV文件路径！")
                return

            # 确保输出目录存在
            output_csv_dir = os.path.dirname(output_csv)
            if output_csv_dir and not os.path.exists(output_csv_dir):
                try:
                    os.makedirs(output_csv_dir)
                except:
                    QMessageBox.warning(self, "目录错误", "无法创建输出CSV文件所在的目录！")
                    return

            if not output_image:
                QMessageBox.warning(self, "输入错误", "请指定输出图像文件夹路径！")
                return

            # 确保输出图像目录存在
            if not os.path.exists(output_image):
                try:
                    os.makedirs(output_image)
                except:
                    QMessageBox.warning(self, "目录错误", "无法创建输出图像文件夹！")
                    return

            # 获取文件夹中的图像数量
            image_files = [f for f in os.listdir(input_folder)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            image_count = len(image_files)

            # 如果没有指定任务名，使用文件夹名
            if not task_name:
                task_name = os.path.basename(input_folder)

            # 创建任务
            task_info = {
                'name': task_name,
                'input_folder': input_folder,
                'output_csv': output_csv,
                'output_image': output_image,
                'image_count': image_count
            }

            # 添加到列表
            task_item = QListWidgetItem(f"{task_name} ({image_count}张图像)")
            task_item.setData(Qt.UserRole, task_info)
            self.batch_tasks_list.addItem(task_item)

            # 更新日志
            self.batch_log.append(f"[{time.strftime('%H:%M:%S')}] 已添加任务: {task_name}, 包含{image_count}张图像")

    def remove_batch_task(self):
        """移除选中的批量处理任务"""
        selected_items = self.batch_tasks_list.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            task_info = item.data(Qt.UserRole)
            row = self.batch_tasks_list.row(item)
            self.batch_tasks_list.takeItem(row)

            # 更新日志
            self.batch_log.append(f"[{time.strftime('%H:%M:%S')}] 已移除任务: {task_info['name']}")

    def start_batch_processing(self):
        """开始批量处理所有任务"""
        # 检查任务列表
        if self.batch_tasks_list.count() == 0:
            QMessageBox.warning(self, "无任务", "请先添加处理任务！")
            return

        # 检查是否已经在处理
        if self.batch_processing:
            QMessageBox.information(self, "处理中", "批量处理已在进行中！")
            return

        # 获取通用设置
        model_path = self.batch_model_path_edit.text()
        if not model_path:
            QMessageBox.warning(self, "模型错误", "请指定YOLOv8模型路径！")
            return
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "模型错误", "指定的模型文件不存在！")
            return

        confidence_threshold = self.batch_confidence_slider.value() / 100.0
        pixel_size_um = float(self.batch_pixel_size_edit.text())
        save_images = self.batch_save_images_check.isChecked()

        # 准备批处理
        self.batch_processing = True
        self.current_batch_task = 0
        self.batch_progress.setValue(0)
        self.current_task_label.setText("准备处理...")
        self.batch_log.clear()
        self.batch_log.append(f"[{time.strftime('%H:%M:%S')}] 开始批量处理 {self.batch_tasks_list.count()} 个任务...")

        # 更新UI状态
        self.start_batch_btn.setEnabled(False)
        self.stop_batch_btn.setEnabled(True)
        self.add_task_btn.setEnabled(False)
        self.remove_task_btn.setEnabled(False)

        # 处理第一个任务
        self._process_next_batch_task(model_path, confidence_threshold, pixel_size_um, save_images)

    def _process_next_batch_task(self, model_path, confidence_threshold, pixel_size_um, save_images):
        """处理下一个批处理任务"""
        if not self.batch_processing or self.current_batch_task >= self.batch_tasks_list.count():
            # 所有任务完成
            self.batch_processing = False
            self.batch_progress.setValue(100)
            self.current_task_label.setText("所有任务已完成！")
            self.batch_log.append(f"[{time.strftime('%H:%M:%S')}] 批量处理完成！")

            # 恢复UI状态
            self.start_batch_btn.setEnabled(True)
            self.stop_batch_btn.setEnabled(False)
            self.add_task_btn.setEnabled(True)
            self.remove_task_btn.setEnabled(True)

            QMessageBox.information(self, "处理完成", "所有批量处理任务已完成！")
            return

        # 获取当前任务
        task_item = self.batch_tasks_list.item(self.current_batch_task)
        task_info = task_item.data(Qt.UserRole)

        # 更新进度
        progress_pct = int((self.current_batch_task / self.batch_tasks_list.count()) * 100)
        self.batch_progress.setValue(progress_pct)
        self.current_task_label.setText(
            f"正在处理: {task_info['name']} ({self.current_batch_task + 1}/{self.batch_tasks_list.count()})")

        # 日志
        self.batch_log.append(f"[{time.strftime('%H:%M:%S')}] 开始处理任务: {task_info['name']}")

        # 创建并启动处理线程
        thread = ProcessThread(
            model_path=model_path,
            image_folder=task_info['input_folder'],
            output_csv=task_info['output_csv'],
            output_image_folder=task_info['output_image'],
            pixel_size_um=pixel_size_um,
            confidence_threshold=confidence_threshold,
            save_images=save_images,
            min_area=self.min_area_spin.value(),
            min_perimeter=self.min_perimeter_spin.value(),
            min_circularity=self.min_circularity_spin.value(),
            max_aspect_ratio=self.max_aspect_ratio_spin.value(),
            edge_detection_mode="strict" if self.edge_detection_combo.currentIndex() == 0 else "relaxed"
        )

        # 连接信号
        thread.progress_signal.connect(lambda p, msg: self._update_batch_task_progress(task_info['name'], p, msg))
        thread.log_signal.connect(lambda msg: self._update_batch_task_log(task_info['name'], msg))
        thread.done_signal.connect(lambda df: self._batch_task_done(df, task_info['name']))

        # 将线程保存到列表中(防止被垃圾回收)
        self.batch_threads.append(thread)

        # 启动线程
        thread.start()

    def _update_batch_task_progress(self, task_name, progress, message):
        """更新批处理任务进度"""
        if not self.batch_processing:
            return

        self.current_task_label.setText(f"正在处理: {task_name} - {message} ({progress}%)")

    def _update_batch_task_log(self, task_name, message):
        """更新批处理任务日志"""
        if not self.batch_processing:
            return

        self.batch_log.append(f"[{time.strftime('%H:%M:%S')}] {task_name}: {message}")

    def resource_path(relative_path):
        """获取资源的绝对路径，兼容开发环境和PyInstaller打包环境"""
        try:
            # PyInstaller创建临时文件夹，将路径存储在_MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            # 非打包环境下，使用当前路径
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def _batch_task_done(self, df, task_name):
        """批处理任务完成"""
        if not self.batch_processing:
            return

        # 记录完成
        self.batch_log.append(f"[{time.strftime('%H:%M:%S')}] 任务完成: {task_name}, 处理了 {len(df)} 个絮团")

        # 移动到下一个任务
        self.current_batch_task += 1

        # 设置任务项的背景颜色为绿色表示已完成
        task_item = self.batch_tasks_list.item(self.current_batch_task - 1)
        task_item.setBackground(QBrush(QColor("#AAFFAA")))

        # 处理下一个
        QTimer.singleShot(500, lambda: self._process_next_batch_task(
            self.batch_model_path_edit.text(),
            self.batch_confidence_slider.value() / 100.0,
            float(self.batch_pixel_size_edit.text()),
            self.batch_save_images_check.isChecked()
        ))

    def stop_batch_processing(self):
        """停止批量处理"""
        if not self.batch_processing:
            return

        # 确认
        reply = QMessageBox.question(
            self, "确认停止",
            "确定要停止当前批量处理任务吗?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # 停止处理
        self.batch_processing = False

        # 停止当前线程
        if self.batch_threads and len(self.batch_threads) > 0:
            current_thread = self.batch_threads[-1]
            if current_thread.isRunning():
                current_thread.stop()

        # 更新UI
        self.current_task_label.setText("批处理已停止")
        self.batch_log.append(f"[{time.strftime('%H:%M:%S')}] 批量处理已手动停止")

        # 恢复UI状态
        self.start_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        self.add_task_btn.setEnabled(True)
        self.remove_task_btn.setEnabled(True)

    # 定义缺少的文件和文件夹浏览方法
    def _browse_folder(self, line_edit, title):
        """浏览文件夹并设置到行编辑控件"""
        try:
            folder = QFileDialog.getExistingDirectory(self, title, "")
            if folder:
                line_edit.setText(folder)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"浏览文件夹时出错:\n{str(e)}")
            import traceback
            print(traceback.format_exc())

    def _browse_save_file(self, line_edit, title, file_filter):
        """浏览保存文件并设置到行编辑控件"""
        try:
            # 尝试获取一个有意义的默认文件名
            default_dir = ""
            default_filename = ""

            # 如果是输出CSV，设置更好的默认名
            if "CSV" in title or "csv" in file_filter:
                # 获取当前日期时间作为文件名的一部分
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                default_filename = f"floc_properties_{timestamp}.csv"

                # 如果是批量处理中的任务，尝试从输入文件夹获取更有意义的名称
                if hasattr(self, 'batch_tasks_list') and title == "选择输出CSV文件":
                    try:
                        # 获取当前输入框中的文本
                        input_folder = None
                        dialog = line_edit.window()
                        for child in dialog.findChildren(QLineEdit):
                            if child.placeholderText() == "输入图像文件夹路径" and child.text():
                                input_folder = child.text()
                                break

                        if input_folder:
                            folder_name = os.path.basename(input_folder)
                            default_filename = f"floc_properties_{folder_name}.csv"
                            default_dir = os.path.dirname(input_folder)  # 默认使用与输入相同的目录
                    except:
                        pass

            # 打开文件对话框，设置默认文件名
            default_path = os.path.join(default_dir, default_filename) if default_dir else default_filename
            file_path, _ = QFileDialog.getSaveFileName(self, title, default_path, file_filter)

            if file_path:
                # 确保CSV文件有正确的扩展名
                if "CSV" in title or "csv" in file_filter:
                    if not file_path.lower().endswith('.csv'):
                        file_path += '.csv'

                line_edit.setText(file_path)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"浏览文件时出错:\n{str(e)}")
            import traceback
            print(traceback.format_exc())

    # F1快捷键连接到帮助功能
    def setup_help_shortcut(self):
        """设置F1快捷键连接到帮助功能"""
        self.help_shortcut = QShortcut(QKeySequence("F1"), self)
        self.help_shortcut.activated.connect(self.show_help)

    def show_help(self):
        """显示帮助文档"""
        # 使用外部浏览器打开帮助文件
        help_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "teaching.html")

        # 如果文件不存在，尝试创建
        if not os.path.exists(help_path):
            # 使用HELP_HTML_CONTENT创建文件
            try:
                with open(help_path, 'w', encoding='utf-8') as f:
                    f.write(HELP_HTML_CONTENT)
                self.log_message("已创建帮助文件")
            except Exception as e:
                QMessageBox.warning(self, "帮助文件创建失败",
                                  f"无法创建帮助文件，请检查磁盘空间和权限。\n错误: {str(e)}")
                return

        # 打开帮助文件
        if os.path.exists(help_path):
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(help_path))
            self.statusBar().showMessage("已在外部浏览器中打开帮助文档")
        else:
            QMessageBox.warning(self, "帮助文件未找到",
                              "无法找到帮助文件，请确保teaching.html文件位于程序目录中。")

    def show_about(self):
        """显示关于对话框"""
        about_text = """
        <h2 style='color: #D81B60;'>絮团形态分析系统</h2>
        <p>版本: 1.0.0</p>
        <p>基于YOLOv8的絮团检测、测量与分析软件</p>
        <p>本软件由 Ya Wu 开发，训练数据由 Ya Wu 和 Ying Chen 共同标注</p>
        """
        QMessageBox.information(self, "关于絮团形态分析系统", about_text)


def closeEvent(self, event):
    """窗口关闭事件"""
    # 如果有Web视图，确保关闭前停止加载
    if WEB_ENGINE_AVAILABLE:
        self.web_view.stop()
    event.accept()


if __name__ == "__main__":
    # 检查是否有中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
        plt.rcParams['axes.unicode_minus'] = False

        # 测试中文字体
        fig = plt.figure(figsize=(1, 1))
        plt.text(0.5, 0.5, '测试中文', fontsize=12)
        plt.close(fig)
    except Exception as e:
        print(f"中文字体设置可能有问题: {e}")
        try:
            # 尝试使用系统字体
            font_manager.fontManager.addfont('/System/Library/Fonts/PingFang.ttc')  # macOS
            # Windows 可能在以下位置
            for font_path in [
                'C:/Windows/Fonts/simhei.ttf',
                'C:/Windows/Fonts/msyh.ttc',
                'C:/Windows/Fonts/simsun.ttc',
            ]:
                if os.path.exists(font_path):
                    font_manager.fontManager.addfont(font_path)

            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'PingFang HK', 'sans-serif']
        except Exception:
            print("无法设置中文字体，图表中的中文可能无法正确显示")

    # 创建应用程序
    app = QApplication(sys.argv)

    # Windows平台专用字体设置
    if sys.platform == 'win32':
        font = QFont("Microsoft YaHei UI", 9)
        app.setFont(font)

    # 创建主窗口
    main_window = MainWindow()
    main_window.show()

    # 运行应用程序
    sys.exit(app.exec_())
