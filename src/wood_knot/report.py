from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


def _paragraph(text: str, style: str | None = None) -> str:
    escaped = escape(text)
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t>{escaped}</w:t></w:r></w:p>"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    def cell(text: str, bold: bool = False) -> str:
        run_props = "<w:rPr><w:b/></w:rPr>" if bold else ""
        return (
            "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
            f"<w:p><w:r>{run_props}<w:t>{escape(text)}</w:t></w:r></w:p></w:tc>"
        )

    header_xml = "".join(cell(header, bold=True) for header in headers)
    row_xml = [f"<w:tr>{header_xml}</w:tr>"]
    for row in rows:
        row_xml.append("<w:tr>" + "".join(cell(value) for value in row) + "</w:tr>")
    return (
        "<w:tbl><w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"4\" w:color=\"999999\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:color=\"999999\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:color=\"999999\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:color=\"999999\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:color=\"999999\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:color=\"999999\"/>"
        "</w:tblBorders></w:tblPr>"
        + "".join(row_xml)
        + "</w:tbl>"
    )


def _document_xml(elements: list[str]) -> str:
    body = "".join(elements)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:rPr><w:b/><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:sz w:val="36"/></w:rPr>
    <w:pPr><w:jc w:val="center"/><w:spacing w:after="260"/></w:pPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="Heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:rPr><w:b/><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:sz w:val="28"/></w:rPr>
    <w:pPr><w:spacing w:before="260" w:after="140"/></w:pPr>
  </w:style>
</w:styles>
"""


def write_docx_report(summary: dict[str, dict[str, float]], output_path: Path) -> None:
    rule = summary["rule"]
    baseline = summary["baseline"]
    elements = [
        _paragraph("《机器视觉与图像处理》课程设计报告", "Title"),
        _paragraph("题目：基于 OpenCV 的木材结疤检测系统", "Title"),
        _paragraph("一、研究对象与性能指标", "Heading1"),
        _paragraph(
            "本设计面向板材分级与加工前检测场景，目标是在木材表面图像中定位结疤区域，同时抑制树心区域误检。"
            "系统以矩形框输出结疤位置，并以 precision、recall、F1 和平均 IoU 评价检测效果。"
        ),
        _paragraph("二、系统硬件选型与架构", "Heading1"),
        _table(
            ["模块", "选型", "作用"],
            [
                ["工业相机", "500 万像素全局快门彩色相机", "采集板材表面纹理与颜色信息"],
                ["镜头", "12 mm 定焦低畸变镜头", "保证视场覆盖和边缘成像质量"],
                ["光源", "条形漫反射 LED，色温 5000 K", "削弱木纹高光，提高结疤对比度"],
                ["计算单元", "工控机 + Python/OpenCV", "完成预处理、候选分割、连通域分析和结果输出"],
                ["输送与触发", "编码器、光电传感器、PLC", "保证采集位置稳定，并与生产线联动"],
            ],
        ),
        _paragraph("三、软件架构与关键算法", "Heading1"),
        _paragraph(
            "软件流程包括图像采集、颜色空间转换、候选区域分割、形态学滤波、连通域筛选、树心排除、结果标注和统计评估。"
            "关键算法先在 HSV 与灰度空间中提取暗棕色候选区域，再用开闭运算去噪和补洞，随后按面积、长宽比、填充率筛除木纹噪声。"
            "树心区域在示例和测试集中表现为蓝色或偏蓝标注，算法通过色相与饱和度规则排除该类区域，避免把树心误报为结疤。"
        ),
        _paragraph("四、程序实现", "Heading1"),
        _table(
            ["文件", "说明"],
            [
                ["src/wood_knot/generator.py", "生成带结疤和树心标注的可控合成板材图像"],
                ["src/wood_knot/detector.py", "OpenCV 规则检测器与灰度阈值基线检测器"],
                ["src/wood_knot/evaluation.py", "IoU、precision、recall、F1 等评价指标"],
                ["scripts/run_experiment.py", "批量生成图像、运行对比实验、输出图表和报告"],
                ["tests/test_detector.py", "核心行为的自动化测试"],
            ],
        ),
        _paragraph("五、验证与现有方法对比", "Heading1"),
        _table(
            ["方法", "Precision", "Recall", "F1", "Mean IoU"],
            [
                [
                    "OpenCV 颜色-形态学规则法",
                    f"{rule['precision']:.3f}",
                    f"{rule['recall']:.3f}",
                    f"{rule['f1']:.3f}",
                    f"{rule['mean_iou']:.3f}",
                ],
                [
                    "灰度阈值基线法",
                    f"{baseline['precision']:.3f}",
                    f"{baseline['recall']:.3f}",
                    f"{baseline['f1']:.3f}",
                    f"{baseline['mean_iou']:.3f}",
                ],
            ],
        ),
        _paragraph(
            "对比结果表明，单纯灰度阈值容易把树心、深色木纹或阴影一起检出；加入颜色约束、形态学和连通域筛选后，"
            "在本设计的验证集上能够提高误检抑制能力。详细样例图、CSV 和 JSON 结果见 outputs 目录。"
        ),
        _paragraph("六、算法使用边界", "Heading1"),
        _table(
            ["边界条件", "影响", "处理建议"],
            [
                ["光照剧烈变化或强反光", "颜色阈值漂移，结疤和背景对比下降", "固定漫反射光源，增加白平衡和亮度归一化"],
                ["结疤颜色接近正常木纹", "候选区域可能漏检", "加入纹理特征或训练式分类器"],
                ["树心颜色不是蓝色标注而接近结疤", "规则排除能力下降", "补充形状、位置和纹理判据"],
                ["木材品种变化大", "阈值需要重新标定", "按木种维护参数或使用小样本校准"],
            ],
        ),
        _paragraph("七、课程要求对应关系", "Heading1"),
        _table(
            ["要求", "对应成果"],
            [
                ["建立系统硬件选型与架构", "报告第二节给出相机、镜头、光源、工控机、触发系统选型"],
                ["建立系统软件架构并设计关键算法", "报告第三节和 detector.py 给出处理流程和关键规则"],
                ["采用 OpenCV 等工具编程实现", "detector.py 使用 cv2 完成颜色空间转换、形态学和连通域分析"],
                ["验证、对比分析、给出算法边界", "run_experiment.py 输出指标、对比图，报告第五和第六节说明边界"],
            ],
        ),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", ZIP_DEFLATED) as docx:
        docx.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
""",
        )
        docx.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
""",
        )
        docx.writestr(
            "word/_rels/document.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
""",
        )
        docx.writestr("word/styles.xml", _styles_xml())
        docx.writestr("word/document.xml", _document_xml(elements))
