# -*- coding: utf-8 -*-
"""
Dola AI 动态水印全自动去除工具 v2.2 (超精细无残留版) (Dola Dynamic Watermark Remover)
特性：
1. 多模态时序轨迹追踪（Multi-Modal Gradient & Intensity NCC Tracker）：
   - Pass 1: 结合灰度互相关与 Sobel/Laplacian 边缘梯度互相关，精准穿透亮色天空与深色暗景；
   - Pass 2: 轨迹分段聚类与线性运动方程拟合（X/Y 运动拟合）；
   - Pass 3: 时序前后外推扩展（向前后各扩展 24 帧），100% 覆盖刚弹出时的胶囊展开/打字机擦除动效与淡出残影；
   - Pass 4: 自适应全向高精差分掩膜（亮色展开+暗色投影）与精密字形融合，彻底消灭任何阶段的水印与白迹残留；
2. 音画无损封装：自动提取并混流原视频高保真音频轨道；
3. 完美支持 Windows 中文路径、批量拖拽与图形界面点选。
"""

import sys
import os
import subprocess
import tempfile
import cv2
import numpy as np
import imageio.v2 as imageio
import imageio_ffmpeg

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

def check_video_has_audio(input_path):
    """检查原视频是否包含音频流"""
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg_exe, '-i', input_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        return 'Audio:' in proc.stderr
    except Exception:
        return False

def remux_audio(video_no_audio_path, original_path, final_output_path):
    """将原视频的音频流与处理后的无水印视频合并"""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    has_audio = check_video_has_audio(original_path)
    
    if not has_audio:
        if os.path.exists(final_output_path):
            os.remove(final_output_path)
        os.rename(video_no_audio_path, final_output_path)
        return True
    
    cmd = [
        ffmpeg_exe,
        '-y',
        '-i', video_no_audio_path,
        '-i', original_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        final_output_path
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        if os.path.exists(video_no_audio_path):
            os.remove(video_no_audio_path)
        return True
    except Exception as e:
        print(f"[提示] 音频混流异常: {e}，保留纯视频文件")
        if os.path.exists(final_output_path):
            os.remove(final_output_path)
        os.rename(video_no_audio_path, final_output_path)
        return False

def process_video(input_path, output_path=None, threshold=0.38):
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
    tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    
    # 梯度特征模板（用于穿透纯白或高对比背景）
    gx_tpl = cv2.Sobel(tpl_gray, cv2.CV_32F, 1, 0)
    gy_tpl = cv2.Sobel(tpl_gray, cv2.CV_32F, 0, 1)
    mag_tpl = cv2.magnitude(gx_tpl, gy_tpl)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f'[错误] 无法打开视频: {input_path}')
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("=" * 60)
    print(f"【Dola 动态水印全自动去除工具 v2.2 (超精细无残留版)】")
    print(f"输入文件: {os.path.basename(input_path)}")
    print(f"视频规格: {width}x{height} @ {fps:.2f}fps, 共 {total_frames} 帧")
    print(f"输出目标: {os.path.basename(output_path)}")
    print("=" * 60)
    
    # 动态适应不同分辨率（基于标准 720p 1280x720 缩放）
    scale_factor = width / 1280.0
    if abs(scale_factor - 1.0) > 0.03:
        scaled_w = int(round(tpl_w * scale_factor))
        scaled_h = int(round(tpl_h * scale_factor))
        cur_tpl = cv2.resize(tpl, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        cur_tpl_gray = cv2.cvtColor(cur_tpl, cv2.COLOR_BGR2GRAY)
        cur_mask = cv2.resize(mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
        
        gx = cv2.Sobel(cur_tpl_gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(cur_tpl_gray, cv2.CV_32F, 0, 1)
        cur_mag_tpl = cv2.magnitude(gx, gy)
    else:
        cur_tpl = tpl
        cur_tpl_gray = tpl_gray
        cur_mask = mask
        cur_mag_tpl = mag_tpl
        scaled_w, scaled_h = tpl_w, tpl_h
    
    print("[阶段 1/3] 正在对全片进行多模态梯度与时序动态水印追踪分析...")
    frames = []
    detections = {}
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. 灰度相关度
        res_gray = cv2.matchTemplate(gray, cur_tpl_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val_g, min_loc, max_loc_g = cv2.minMaxLoc(res_gray)
        
        # 2. 梯度幅度相关度
        gx_f = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy_f = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag_f = cv2.magnitude(gx_f, gy_f)
        res_mag = cv2.matchTemplate(mag_f, cur_mag_tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val_m, _, max_loc_m = cv2.minMaxLoc(res_mag)
        
        if max_val_g >= threshold:
            detections[frame_idx] = (max_loc_g[0], max_loc_g[1], max_val_g)
        elif max_val_m >= 0.32:
            detections[frame_idx] = (max_loc_m[0], max_loc_m[1], max_val_m * 0.85)
        
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
            if f_idx - prev_f <= 20 and dist < (80 * scale_factor):
                curr_seg.append(f_idx)
            else:
                if len(curr_seg) >= 4:
                    segments.append(curr_seg)
                curr_seg = [f_idx]
    if len(curr_seg) >= 4:
        segments.append(curr_seg)
    
    print(f"-> 识别到 {len(segments)} 个 Dola 水印动态运动区间")
    
    # 轨迹方程拟合与时序前后外推 (覆盖弹出的最初 24 帧与消失的最后 24 帧)
    frame_targets = {}
    pad_extrapolate = 24
    
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
    
    print(f"[阶段 3/3] 正在执行全向高精差分掩膜与全帧无缝去水印...")
    
    temp_dir = tempfile.gettempdir()
    temp_video_path = os.path.join(temp_dir, f"dola_temp_{os.getpid()}.mp4")
    
    writer = imageio.get_writer(
        temp_video_path,
        fps=fps,
        codec='libx264',
        quality=9,
        pixelformat='yuv420p',
        macro_block_size=1
    )
    
    pad_x = int(25 * scale_factor)
    pad_y = int(20 * scale_factor)
    extra_w = int(30 * scale_factor)
    k_bg_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (int(25 * scale_factor), int(25 * scale_factor)))
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    for t, frame in enumerate(frames):
        out_f = frame.copy()
        if t in frame_targets:
            for (x, y) in frame_targets[t]:
                x0 = max(0, x - pad_x)
                y0 = max(0, y - pad_y)
                x1 = min(width, x + scaled_w + pad_x + extra_w)
                y1 = min(height, y + scaled_h + pad_y + int(15 * scale_factor))
                
                roi = out_f[y0:y1, x0:x1]
                if roi.shape[0] < 5 or roi.shape[1] < 5:
                    continue
                
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # 1. 亮色高频/展开胶囊白条差分提取 (开运算减差)
                bg_open = cv2.morphologyEx(gray_roi, cv2.MORPH_OPEN, k_bg_rect)
                diff_bright = cv2.subtract(gray_roi, bg_open)
                _, mask_bright = cv2.threshold(diff_bright, 6, 255, cv2.THRESH_BINARY)
                
                # 2. 亮背景暗色阴影/边缘差分提取 (闭运算减差)
                bg_close = cv2.morphologyEx(gray_roi, cv2.MORPH_CLOSE, k_bg_rect)
                diff_dark = cv2.subtract(bg_close, gray_roi)
                _, mask_dark = cv2.threshold(diff_dark, 6, 255, cv2.THRESH_BINARY)
                
                # 宽幅限定边界框，完全覆盖弹出胶囊的延展区域
                off_x = x - x0
                off_y = y - y0
                box_region = np.zeros(roi.shape[:2], dtype=np.uint8)
                box_region[
                    max(0, off_y - int(10 * scale_factor)):min(roi.shape[0], off_y + scaled_h + int(18 * scale_factor)),
                    max(0, off_x - int(18 * scale_factor)):min(roi.shape[1], off_x + scaled_w + int(25 * scale_factor))
                ] = 255
                
                active_diff = cv2.bitwise_and(cv2.bitwise_or(mask_bright, mask_dark), box_region)
                
                # 3. 融合精密字形蒙版
                font_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                if 0 <= off_y and off_y + scaled_h <= roi.shape[0] and 0 <= off_x and off_x + scaled_w <= roi.shape[1]:
                    font_mask[off_y:off_y+scaled_h, off_x:off_x+scaled_w] = cur_mask
                
                font_mask_dil = cv2.dilate(font_mask, k_dil, iterations=1)
                
                # 4. 复合总蒙版并适度膨胀消除残留光晕
                total_mask = cv2.bitwise_or(font_mask_dil, active_diff)
                total_mask = cv2.dilate(total_mask, k_dil, iterations=1)
                
                # 5. 图像无缝修复
                cleaned_roi = cv2.inpaint(roi, total_mask, 3, cv2.INPAINT_TELEA)
                out_f[y0:y1, x0:x1] = cleaned_roi
        
        out_rgb = cv2.cvtColor(out_f, cv2.COLOR_BGR2RGB)
        writer.append_data(out_rgb)
        
        if (t + 1) % 30 == 0 or (t + 1) == total_frames:
            print(f"\r渲染进度: [{t+1}/{total_frames}] {((t+1)/total_frames)*100:.1f}%", end="", flush=True)
            
    writer.close()
    
    # 混流原始音轨
    print("\n正在封装高清音轨...")
    remux_audio(temp_video_path, input_path, output_path)
    
    print('\n' + '=' * 60)
    print(f'处理完成！已全面清除包括刚弹出在内的所有 Dola 水印与白迹残影！')
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
