# -*- coding: utf-8 -*-
"""
豆包 AI (Doubao) 动态水印全自动去除工具 v2.0 (Doubao Dynamic Watermark Remover)
特性：
1. 多模态时序轨迹追踪（Multi-Modal Gradient & Intensity NCC Tracker）：
   - Pass 1: 结合灰度互相关与 Sobel/Laplacian 边缘梯度互相关，精准穿透亮色天空与深色暗景；
   - Pass 2: 区域聚焦检测与轨迹分段聚类；
   - Pass 3: 运动方程线性拟合（X/Y Drift Fitting）与时序前后外推扩展（前后各外推 18 帧），彻底消除淡入淡出动效残影；
   - Pass 4: 复合字形/阴影掩膜与无缝图像修复（Inpainting）；
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
    for sub in ['doubao_assets', 'assets']:
        tpl_path = os.path.join(script_dir, sub, 'tpl_precise.png')
        mask_path = os.path.join(script_dir, sub, 'mask_precise.png')
        
        if os.path.exists(tpl_path) and os.path.exists(mask_path):
            tpl = cv2_imread(tpl_path, cv2.IMREAD_COLOR)
            mask = cv2_imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if tpl is not None and mask is not None:
                return tpl, mask
    
    raise FileNotFoundError('未找到豆包标定模板与蒙版文件！请确保 doubao_assets 目录完整。')

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

def process_video(input_path, output_path=None, threshold=0.42):
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
    print(f"【豆包 AI 视频水印全自动去除工具 v2.0】")
    print(f"输入文件: {os.path.basename(input_path)}")
    print(f"视频规格: {width}x{height} @ {fps:.2f}fps, 共 {total_frames} 帧")
    print(f"输出目标: {os.path.basename(output_path)}")
    print("=" * 60)
    
    # 动态适应不同分辨率（基于标准 720x1280 竖屏标定）
    base_dim = min(width, height)
    scale_factor = base_dim / 720.0
    
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
    
    # 定义四个角落重点候选 ROI 区域（加速并降低误检）
    margin_w = int(scaled_w * 1.8)
    margin_h = int(scaled_h * 2.8)
    
    rois_def = {
        'TL': (0, 0, min(width, margin_w), min(height, margin_h)),
        'TR': (max(0, width - margin_w), 0, width, min(height, margin_h)),
        'BL': (0, max(0, height - margin_h), min(width, margin_w), height),
        'BR': (max(0, width - margin_w), max(0, height - margin_h), width, height),
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        best_score = -1.0
        best_loc = None
        
        for r_name, (rx0, ry0, rx1, ry1) in rois_def.items():
            roi_g = gray[ry0:ry1, rx0:rx1]
            if roi_g.shape[0] < scaled_h or roi_g.shape[1] < scaled_w:
                continue
            
            # 1. 灰度相关度
            res_gray = cv2.matchTemplate(roi_g, cur_tpl_gray, cv2.TM_CCOEFF_NORMED)
            _, max_v_g, _, max_l_g = cv2.minMaxLoc(res_gray)
            
            # 2. 梯度幅度相关度（在强光/浅色背景下大幅提升准确度）
            gx_roi = cv2.Sobel(roi_g, cv2.CV_32F, 1, 0)
            gy_roi = cv2.Sobel(roi_g, cv2.CV_32F, 0, 1)
            mag_roi = cv2.magnitude(gx_roi, gy_roi)
            res_mag = cv2.matchTemplate(mag_roi, cur_mag_tpl, cv2.TM_CCOEFF_NORMED)
            _, max_v_m, _, max_l_m = cv2.minMaxLoc(res_mag)
            
            # 综合评分判定
            if max_v_g >= threshold:
                abs_x = rx0 + max_l_g[0]
                abs_y = ry0 + max_l_g[1]
                if max_v_g > best_score:
                    best_score = max_v_g
                    best_loc = (abs_x, abs_y)
            elif max_v_m >= 0.38:
                abs_x = rx0 + max_l_m[0]
                abs_y = ry0 + max_l_m[1]
                eff_score = max(max_v_g, max_v_m * 0.85)
                if eff_score > best_score:
                    best_score = eff_score
                    best_loc = (abs_x, abs_y)
        
        if best_loc is not None and best_score >= threshold:
            detections[frame_idx] = (best_loc[0], best_loc[1], best_score)
        
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
            if f_idx - prev_f <= 25 and dist < (80 * scale_factor):
                curr_seg.append(f_idx)
            else:
                if len(curr_seg) >= 4:
                    segments.append(curr_seg)
                curr_seg = [f_idx]
    if len(curr_seg) >= 4:
        segments.append(curr_seg)
    
    print(f"-> 识别到 {len(segments)} 个豆包水印动态运动区间")
    
    # 轨迹方程拟合与时序前后外推 (覆盖刚弹出与淡出的前后 18 帧)
    frame_targets = {}
    pad_extrapolate = 18
    
    for s_idx, seg in enumerate(segments):
        t_vals = np.array(seg)
        x_vals = np.array([detections[t][0] for t in seg])
        y_vals = np.array([detections[t][1] for t in seg])
        
        # 一阶线性运动拟合
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
    
    temp_dir = tempfile.gettempdir()
    temp_video_path = os.path.join(temp_dir, f"doubao_temp_{os.getpid()}.mp4")
    
    writer = imageio.get_writer(
        temp_video_path,
        fps=fps,
        codec='libx264',
        quality=9,
        pixelformat='yuv420p',
        macro_block_size=1
    )
    
    pad_x = int(8 * scale_factor)
    pad_y = int(8 * scale_factor)
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    for t, frame in enumerate(frames):
        out_f = frame.copy()
        if t in frame_targets:
            for (x, y) in frame_targets[t]:
                x0 = max(0, x - pad_x)
                y0 = max(0, y - pad_y)
                x1 = min(width, x + scaled_w + pad_x)
                y1 = min(height, y + scaled_h + pad_y)
                
                roi = out_f[y0:y1, x0:x1]
                if roi.shape[0] < 5 or roi.shape[1] < 5:
                    continue
                
                # 构建精确蒙版
                local_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                off_x = x - x0
                off_y = y - y0
                
                if 0 <= off_y and off_y + scaled_h <= roi.shape[0] and 0 <= off_x and off_x + scaled_w <= roi.shape[1]:
                    local_mask[off_y:off_y+scaled_h, off_x:off_x+scaled_w] = cur_mask
                
                # 适度膨胀
                local_mask = cv2.dilate(local_mask, kernel_dil, iterations=1)
                
                # 图像无缝修复 (Navier-Stokes / Telea)
                cleaned_roi = cv2.inpaint(roi, local_mask, 5, cv2.INPAINT_TELEA)
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
    print(f'处理完成！已全面清除包括淡入淡出在内的所有豆包水印帧！')
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
            title="选择要去除水印的豆包 (Doubao) 视频文件",
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
            default_test = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doubao.mp4")
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
