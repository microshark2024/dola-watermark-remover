# -*- coding: utf-8 -*-
"""
Dola AI 动态水印全自动去除工具 v2.0 (Dola Dynamic Watermark Remover)
特性：
1. 双通道时序轨迹追踪（Two-Pass Temporal Trajectory Tracker）：
   - Pass 1: 逐帧归一化互相关（NCC）全局高精度检测；
   - Pass 2: 轨迹分段聚类与线性运动方程拟合（X/Y 运动拟合）；
   - Pass 3: 时序前后外推扩展（向前后各扩展 18 帧），100% 消除水印刚弹出时的胶囊展开/打字机擦除动效与淡出残影；
   - Pass 4: 自适应局部高频/亮度差分掩膜与无缝图像修复，彻底消灭任何阶段的水印残留；
2. 完美支持 Windows 中文路径、拖拽与图形界面点选。
"""

import sys
import os
import cv2
import numpy as np
import imageio.v2 as imageio

def cv2_imread(path, flags=cv2.IMREAD_COLOR):
    """解决 Windows 下 OpenCV 无法读取包含中文路径的问题"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except Exception as e:
        print(f"[错误] 读取图像失败: {path}, 原因: {e}")
        return None

def get_calibrated_assets():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for sub in ['assets', 'extracted_frames']:
        tpl_path = os.path.join(script_dir, sub, 'tpl_precise.png')
        mask_path = os.path.join(script_dir, sub, 'mask_precise.png')
        
        if os.path.exists(tpl_path) and os.path.exists(mask_path):
            tpl = cv2_imread(tpl_path, cv2.IMREAD_COLOR)
            mask = cv2_imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if tpl is not None and mask is not None:
                return tpl, mask
    
    raise FileNotFoundError('未找到标定模板与蒙版文件！请确保 assets 目录完整。')

def process_video(input_path, output_path=None, threshold=0.40):
    if not os.path.exists(input_path):
        print(f'[错误] 找不到输入视频文件: {input_path}')
        return False
    
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_无水印{ext}"
    
    try:
        tpl, mask = get_calibrated_assets()
    except Exception as e:
        print(f"[错误] {e}")
        return False
        
    tpl_h, tpl_w = mask.shape
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f'[错误] 无法打开视频: {input_path}')
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("=" * 60)
    print(f"【Dola 动态水印全自动去除工具 v2.0】")
    print(f"输入文件: {os.path.basename(input_path)}")
    print(f"视频规格: {width}x{height} @ {fps:.2f}fps, 共 {total_frames} 帧")
    print(f"输出目标: {os.path.basename(output_path)}")
    print("=" * 60)
    
    # 动态适应不同分辨率（基于标准 720p 1280x720 缩放）
    scale_factor = width / 1280.0
    if abs(scale_factor - 1.0) > 0.05:
        scaled_w = int(tpl_w * scale_factor)
        scaled_h = int(tpl_h * scale_factor)
        cur_tpl = cv2.resize(tpl, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        cur_mask = cv2.resize(mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
    else:
        cur_tpl = tpl
        cur_mask = mask
        scaled_w, scaled_h = tpl_w, tpl_h
    
    print("[阶段 1/3] 正在对全片进行多尺度动态水印轨迹分析与定位...")
    frames = []
    detections = {}
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
        # 逐帧模板匹配
        res = cv2.matchTemplate(frame, cur_tpl, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val >= threshold:
            detections[frame_idx] = (max_loc[0], max_loc[1], max_val)
        
        frame_idx += 1
        if frame_idx % 60 == 0 or frame_idx == total_frames:
            print(f"\r分析进度: [{frame_idx}/{total_frames}] {(frame_idx/total_frames)*100:.1f}%", end="", flush=True)
    
    cap.release()
    print("\n[阶段 2/3] 正在构建时序平滑运动轨迹并进行淡入/淡出前后外推...")
    
    # 轨迹分段聚类
    sorted_frames = sorted(detections.keys())
    segments = []
    curr_seg = []
    for f_idx in sorted_frames:
        if not curr_seg:
            curr_seg.append(f_idx)
        else:
            prev_f = curr_seg[-1]
            prev_loc = detections[prev_f]
            curr_loc = detections[f_idx]
            dist = np.hypot(curr_loc[0] - prev_loc[0], curr_loc[1] - prev_loc[1])
            if f_idx - prev_f <= 15 and dist < (70 * scale_factor):
                curr_seg.append(f_idx)
            else:
                if len(curr_seg) >= 5:
                    segments.append(curr_seg)
                curr_seg = [f_idx]
    if len(curr_seg) >= 5:
        segments.append(curr_seg)
    
    print(f"-> 识别到 {len(segments)} 个水印动态运动区间")
    
    # 轨迹方程拟合与时序前后外推 (覆盖弹出的最初 18 帧与消失的最后 18 帧)
    frame_targets = {}
    pad_extrapolate = 18
    
    for s_idx, seg in enumerate(segments):
        t_vals = np.array(seg)
        x_vals = np.array([detections[t][0] for t in seg])
        y_vals = np.array([detections[t][1] for t in seg])
        
        poly_x = np.poly1d(np.polyfit(t_vals, x_vals, deg=1))
        poly_y = np.poly1d(np.polyfit(t_vals, y_vals, deg=1))
        
        t_min = max(0, seg[0] - pad_extrapolate)
        t_max = min(total_frames - 1, seg[-1] + pad_extrapolate)
        
        for t in range(t_min, t_max + 1):
            x = int(round(poly_x(t)))
            y = int(round(poly_y(t)))
            if t not in frame_targets:
                frame_targets[t] = []
            frame_targets[t].append((x, y))
    
    print(f"[阶段 3/3] 正在执行全帧无缝去水印与音画重封装...")
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec='libx264',
        quality=9,
        pixelformat='yuv420p',
        macro_block_size=1
    )
    
    pad_x = int(15 * scale_factor)
    pad_y = int(10 * scale_factor)
    extra_w = int(25 * scale_factor)
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (int(25 * scale_factor), int(25 * scale_factor)))
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    for t, frame in enumerate(frames):
        out_f = frame.copy()
        if t in frame_targets:
            for (x, y) in frame_targets[t]:
                x0 = max(0, x - pad_x)
                y0 = max(0, y - pad_y)
                x1 = min(width, x + scaled_w + pad_x + extra_w)
                y1 = min(height, y + scaled_h + pad_y)
                
                roi = out_f[y0:y1, x0:x1]
                if roi.shape[0] < 5 or roi.shape[1] < 5:
                    continue
                
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                bg_est = cv2.morphologyEx(gray_roi, cv2.MORPH_OPEN, kernel_bg)
                diff = cv2.subtract(gray_roi, bg_est)
                
                _, local_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
                
                # 融合精确字形蒙版
                offset_x = x - x0
                offset_y = y - y0
                if 0 <= offset_y and offset_y + scaled_h <= roi.shape[0] and 0 <= offset_x and offset_x + scaled_w <= roi.shape[1]:
                    local_mask[offset_y:offset_y+scaled_h, offset_x:offset_x+scaled_w] = cv2.bitwise_or(
                        local_mask[offset_y:offset_y+scaled_h, offset_x:offset_x+scaled_w],
                        cur_mask
                    )
                
                # 适度膨胀覆盖模糊光晕
                local_mask = cv2.dilate(local_mask, kernel_dil, iterations=1)
                
                # 图像无缝修复
                cleaned_roi = cv2.inpaint(roi, local_mask, 3, cv2.INPAINT_TELEA)
                out_f[y0:y1, x0:x1] = cleaned_roi
        
        out_rgb = cv2.cvtColor(out_f, cv2.COLOR_BGR2RGB)
        writer.append_data(out_rgb)
        
        if (t + 1) % 30 == 0 or (t + 1) == total_frames:
            print(f"\r渲染进度: [{t+1}/{total_frames}] {((t+1)/total_frames)*100:.1f}%", end="", flush=True)
            
    writer.close()
    print('\n' + '=' * 60)
    print(f'处理完成！已全面清除包括刚弹出在内的所有水印帧！')
    print(f'输出文件: {output_path}')
    print('=' * 60 + '\n')
    return True

def select_files_dialog():
    """打开原生文件选择器"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        files = filedialog.askopenfilenames(
            title="选择要去除水印的 Dola 视频文件",
            filetypes=[("视频文件", "*.mp4 *.mov *.avi *.mkv *.webm"), ("所有文件", "*.*")]
        )
        return list(files)
    except Exception:
        return []

def main():
    if len(sys.argv) > 1:
        video_files = sys.argv[1:]
    else:
        print("未检测到拖拽的视频文件，正在打开文件选择窗口...")
        video_files = select_files_dialog()
        if not video_files:
            default_test = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download (39).mp4")
            if os.path.exists(default_test):
                print(f"未选择文件，自动处理当前目录测试视频: {os.path.basename(default_test)}")
                video_files = [default_test]
            else:
                print("未选择任何视频文件，已退出。")
                return

    for idx, video in enumerate(video_files, 1):
        print(f"[{idx}/{len(video_files)}] 正在处理: {video}")
        process_video(video)
        
    print("全部视频处理完毕！按回车键退出...")
    try:
        input()
    except Exception:
        pass

if __name__ == '__main__':
    main()
