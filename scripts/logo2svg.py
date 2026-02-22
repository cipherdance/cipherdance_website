#!/usr/bin/env python3
"""
Logo → 单色 SVG 转换工具 (vtracer + opencv)
用法: python3 logo2svg.py input.jpg output.svg [--height 41] [--color "#FFB072"]

流水线:
  1. 智能检测背景 (亮/暗) 并去除
  2. 双边滤波降噪 (去 JPG 伪影, 保留边缘)
  3. Otsu 自动阈值二值化
  4. 形态学清理 (去碎点, 填小洞)
  5. vtracer spline 矢量化
  6. 重新着色 + 输出透明背景 SVG

依赖: pip install vtracer opencv-python-headless Pillow
"""

import sys
import os
import re
import tempfile
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import vtracer


def _detect_complexity(gray):
    """用 Canny 边缘密度判断图片复杂度 (0~1), 越高=纹理越多"""
    edges = cv2.Canny(gray, 50, 150)
    return np.count_nonzero(edges) / edges.size


def _filter_small_components(binary, min_ratio=0.005):
    """只保留面积 >= 总像素 * min_ratio 的连通域, 丢弃碎片"""
    total = binary.size
    min_area = int(total * min_ratio)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):  # 跳过 0 (背景)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


def logo_to_svg(input_path, output_path, target_height=41, color="#FFB072"):
    """完整流水线: 光栅 Logo → 干净单色 SVG"""

    # ===== 1. 读取图片, 智能检测背景 =====
    print(f"📖 读取: {input_path}")
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        # opencv 不支持 webp 某些情况, 用 PIL 兜底
        pil_img = Image.open(input_path).convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 如果有 alpha 通道, 保留它
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    has_alpha = img.shape[2] == 4
    if has_alpha:
        alpha = img[:, :, 3]
        bgr = img[:, :, :3]
    else:
        bgr = img
        alpha = None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 检测背景: 采样四边 5px 边缘像素
    border = np.concatenate([
        gray[:5, :].flatten(), gray[-5:, :].flatten(),
        gray[:, :5].flatten(), gray[:, -5:].flatten()
    ])
    bg_mean = np.mean(border)
    is_dark_bg = bg_mean < 128
    print(f"   背景: {'暗色' if is_dark_bg else '亮色'} (均值 {bg_mean:.0f})")

    # ===== 1.5 检测图片复杂度 =====
    # 小图 (<200px) 天然边缘密度高, 不算复杂
    complexity = _detect_complexity(gray)
    is_complex = complexity > 0.06 and min(h, w) >= 200
    print(f"   复杂度: {complexity:.3f} ({'复杂图片⚠️' if is_complex else '简单Logo'})")

    # ===== 2. 降噪 (复杂图片用高斯猛模糊, 简单Logo用双边保边) =====
    if is_complex:
        # 复杂图片: 高斯重度模糊, 抹掉纹理只留大轮廓
        blur_radius = max(h, w) // 30  # 自适应: 图片越大模糊越强
        blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        print(f"🔇 重度降噪 (高斯 {blur_radius}px, 抹掉纹理)...")
        gray_clean = cv2.GaussianBlur(gray, (blur_radius, blur_radius), 0)
    else:
        print("🔇 降噪 (双边滤波)...")
        denoised = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
        gray_clean = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    # ===== 3. 二值化 =====
    print("⬛ 二值化 (Otsu)...")
    if is_dark_bg:
        _, binary = cv2.threshold(gray_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray_clean, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 如果有 alpha 通道, 用它精确裁切前景
    if alpha is not None:
        binary[alpha < 128] = 0

    # ===== 4. 形态学清理 =====
    print("🧹 清理...")
    if is_complex:
        # 复杂图片: 大核 + 多次迭代, 彻底抹平碎片
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        # 连通域面积过滤: 只保留大块形状
        binary = _filter_small_components(binary, min_ratio=0.01)
        print("   已过滤小碎片, 只保留主要形状")
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ===== 5. 自动裁切空白 =====
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, cw, ch = cv2.boundingRect(coords)
        # 加 3% padding
        pad_x = max(int(cw * 0.03), 2)
        pad_y = max(int(ch * 0.03), 2)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + cw + pad_x)
        y2 = min(h, y + ch + pad_y)
        binary = binary[y1:y2, x1:x2]
        print(f"✂️  裁切至 {binary.shape[1]}x{binary.shape[0]}")

    # ===== 6. vtracer 矢量化 =====
    print("✏️  矢量化 (vtracer spline)...")

    # 构建黑色 Logo 在白色背景上的 PNG (vtracer binary 模式需要)
    bh, bw = binary.shape
    canvas = np.ones((bh, bw, 3), dtype=np.uint8) * 255
    canvas[binary > 0] = [0, 0, 0]

    # 编码为 PNG bytes (无损, 不会引入新噪点)
    pil_img = Image.fromarray(canvas)
    buf = BytesIO()
    pil_img.save(buf, format='PNG')
    png_bytes = buf.getvalue()

    speckle = 20 if is_complex else 6  # 复杂图片更激进地过滤碎点
    svg_str = vtracer.convert_raw_image_to_svg(
        png_bytes,
        img_format='png',
        colormode='binary',       # 单色模式
        mode='spline',            # 平滑贝塞尔曲线
        filter_speckle=speckle,   # 过滤碎点
        corner_threshold=60,      # 角度阈值
        length_threshold=4.0,     # 最小线段长度
        max_iterations=10,        # 平滑迭代
        splice_threshold=45,      # 拼接阈值
        path_precision=8,         # 坐标精度
    )

    # ===== 7. 换色: 黑→目标色, 白→透明 =====
    print(f"🎨 换色 → {color}")
    svg_str = re.sub(r'fill="#000000"', f'fill="{color}"', svg_str)
    svg_str = re.sub(r'fill="#ffffff"', 'fill="none"', svg_str, flags=re.IGNORECASE)

    # 调整 viewBox / 尺寸 — 按目标高度等比缩放
    vb_match = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg_str)
    if vb_match:
        vb_w, vb_h = int(vb_match.group(1)), int(vb_match.group(2))
        scale = target_height / vb_h
        svg_w = round(vb_w * scale, 2)
        # 替换 width/height, 保留 viewBox 不变
        svg_str = re.sub(r'width="\d+"', f'width="{svg_w}"', svg_str)
        svg_str = re.sub(r'height="\d+"', f'height="{target_height}"', svg_str)

    # ===== 8. 保存 =====
    with open(output_path, 'w') as f:
        f.write(svg_str)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"✅ 完成: {output_path} ({size_kb:.1f} KB)")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("示例:")
        print("  python3 logo2svg.py logo.jpg out.svg")
        print("  python3 logo2svg.py logo.png out.svg --height 60")
        print('  python3 logo2svg.py logo.jpg out.svg --color "#FF6B00"')
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # 解析可选参数
    height = 41
    color = "#FFB072"
    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == '--height' and i + 1 < len(args):
            height = int(args[i + 1]); i += 2
        elif args[i] == '--color' and i + 1 < len(args):
            color = args[i + 1]; i += 2
        else:
            i += 1

    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        sys.exit(1)

    logo_to_svg(input_path, output_path, target_height=height, color=color)


if __name__ == "__main__":
    main()
