# 🎯 FlocAnalyzer - 絮团检测与分析系统

基于 YOLOv8 + PyQt5 的图形化絮团图像识别与分析软件。

## 🧰 功能特色
- 🔍 使用 YOLOv8 进行絮团目标检测与分割
- 📐 自动计算形态学参数（面积、周长、等效直径、圆度、分形维数等）
- 📊 可视化图表展示分析结果（直方图、散点图、箱线图等）
- 📄 自动生成 PDF 报告（包含图像、统计图和数据表）
- 📂 支持批量图像处理

## 🖥️ 使用环境
- Python ≥ 3.8
- 需安装 PyTorch 和 ultralytics 的 YOLOv8
- 建议使用带 GPU 的设备以加速推理

## 📦 安装依赖

```bash
pip install -r requirements.txt
```