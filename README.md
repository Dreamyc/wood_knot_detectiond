<<<<<<< HEAD
# 基于 OpenCV 的木材结疤检测系统

本项目用于完成《机器视觉与图像处理》课程设计要求。系统面向木材表面缺陷检测，目标是在检测结疤区域的同时避免把树心区域误检为结疤。

## 项目结构

- `src/wood_knot/generator.py`：生成带结疤和树心标注的可控合成板材图像。
- `src/wood_knot/detector.py`：OpenCV 颜色-形态学规则检测器，以及灰度阈值基线方法。
- `src/wood_knot/evaluation.py`：IoU、precision、recall、F1 等评价指标。
- `src/wood_knot/visualization.py`：标注图和指标对比图生成。
- `src/wood_knot/report.py`：生成 Word 课程设计报告。
- `scripts/run_experiment.py`：批量生成验证集、运行算法对比并输出报告。
- `tests/test_detector.py`：核心检测与评估逻辑测试。

## 运行方式

```powershell
cd D:\download\home\wood_knot_detection
python -m unittest discover -s tests -v
python scripts\run_experiment.py
```

实验输出位于 `outputs/`：

- `metrics.csv`：每张验证图的检测指标。
- `summary.json`：OpenCV 规则法与灰度阈值基线法的平均指标。
- `comparison_chart.png`：性能对比柱状图。
- `samples/`：输入样例与检测标注样例。
- `课程设计报告-木材结疤检测系统.docx`：课程设计报告。

## 方法概要

规则法使用 OpenCV 完成 RGB 到 HSV/灰度转换，先提取暗棕色候选区域，再进行形态学开闭运算、连通域分析和面积/长宽比/填充率筛选。树心区域通过蓝色色相和饱和度约束被排除。对比基线为单纯灰度阈值方法，用于说明颜色与形态约束对误检抑制的作用。
=======
# wood_knot_detectiond
机器视觉与图像处理大作业
>>>>>>> 5d5b2b2bb741147e72f184d1fe55552bf37c18fa
