#!/usr/bin/env python3
"""
图片转橙色风格 SVG 工具
用法: python3 img2svg.py input.jpg output.svg [scale]

流程:
  1. 转灰度
  2. 高斯模糊 (减少噪点)
  3. 二值化 (黑白两色)
  4. potrace 矢量化
  5. 换色 + 加边框背景
"""

import sys
import os
import re
import subprocess
import tempfile
from PIL import Image, ImageFilter, ImageOps


def img_to_svg(input_path, output_path, scale=0.30):
    """
    将图片转换为橙色风格 SVG

    Args:
        input_path: 输入图片路径 (jpg/png)
        output_path: 输出 SVG 路径
        scale: 图片在 SVG 中的缩放比例 (默认 0.30)
    """

    # 颜色配置
    ORANGE = "#FF6B00"      # 橙色线条/填充
    BACKGROUND = "#FFFDF7"  # 奶白色背景

    # SVG 尺寸配置
    SVG_WIDTH = 388
    SVG_HEIGHT = 305

    # ===== 1. 读取图片 =====
    print(f"📖 读取图片: {input_path}")
    img = Image.open(input_path)

    # ===== 2. 转灰度 =====
    print("🔲 转换为灰度...")
    gray = img.convert('L')

    # ===== 3. 高斯模糊 (减少噪点) =====
    print("🌫️  高斯模糊...")
    gray = gray.filter(ImageFilter.GaussianBlur(radius=1))

    # ===== 4. 二值化 =====
    print("⬛⬜ 二值化...")
    gray = ImageOps.autocontrast(gray)
    threshold = 128
    binary = gray.point(lambda x: 255 if x > threshold else 0)

    # ===== 5. 保存为 PBM (potrace 需要) =====
    with tempfile.NamedTemporaryFile(suffix='.pbm', delete=False) as tmp_pbm:
        pbm_path = tmp_pbm.name
    binary.save(pbm_path)

    # ===== 6. potrace 矢量化 =====
    print("✏️  矢量化...")
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_svg:
        raw_svg_path = tmp_svg.name

    subprocess.run([
        'potrace', pbm_path,
        '-s',           # 输出 SVG
        '-t', '5',      # 容差
        '-O', '1',      # 优化级别
        '-o', raw_svg_path
    ], check=True)

    # ===== 7. 读取 potrace 输出 =====
    with open(raw_svg_path, 'r') as f:
        content = f.read()

    # 提取 <g> 元素
    g_start = content.find('<g transform=')
    g_end = content.rfind('</g>') + 4

    if g_start == -1:
        raise ValueError("无法从 potrace 输出中提取图形数据")

    g_content = content[g_start:g_end]

    # ===== 8. 换色 =====
    print(f"🎨 换色为 {ORANGE}...")
    g_content = g_content.replace('#000000', ORANGE)

    # ===== 9. 组装最终 SVG =====
    print("📦 组装 SVG...")

    # 计算居中偏移 (原图 1080x1080)
    original_size = 1080
    offset_x = (SVG_WIDTH - original_size * scale) / 2
    offset_y = (SVG_HEIGHT - original_size * scale) / 2

    final_svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" fill="none" xmlns="http://www.w3.org/2000/svg">
<defs>
  <clipPath id="roundedClip">
    <rect x="8" y="8" width="371" height="289" rx="15"/>
  </clipPath>
</defs>
<!-- 奶白色背景 -->
<rect x="3" y="3" width="381" height="299" rx="20" fill="{BACKGROUND}"/>
<!-- 橙色边框 -->
<rect x="3" y="3" width="381" height="299" rx="20" stroke="{ORANGE}" stroke-width="5" fill="none"/>
<!-- 图片内容 -->
<g clip-path="url(#roundedClip)">
  <g transform="translate({offset_x:.1f}, {offset_y:.1f}) scale({scale})">
  {g_content}
  </g>
</g>
</svg>'''

    # ===== 10. 保存 =====
    with open(output_path, 'w') as f:
        f.write(final_svg)

    # 清理临时文件
    os.unlink(pbm_path)
    os.unlink(raw_svg_path)

    # 获取文件大小
    file_size = os.path.getsize(output_path)
    print(f"✅ 完成! 输出: {output_path} ({file_size / 1024:.1f} KB)")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("示例:")
        print("  python3 img2svg.py photo.jpg output.svg")
        print("  python3 img2svg.py photo.png output.svg 0.35")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0.30

    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        sys.exit(1)

    img_to_svg(input_path, output_path, scale)


if __name__ == "__main__":
    main()
