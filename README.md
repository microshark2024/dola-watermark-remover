# Dola AI Dynamic Watermark Remover (Dola 动态水印去除工具)

针对 Dola 官网最新上线的全画幅动态漂移水印（半透明 `Dola AI` 字样），基于计算机视觉开发的自动化无缝消除工具。

## ✨ 核心特性

1. **双通道时序轨迹追踪（Two-Pass Temporal Trajectory Tracker）**：
   - 自动逐帧追踪每隔 3 秒切换位置、匀速漂移的动态水印；
   - 自动拟合水印的物理运动轨迹方程。
2. **淡入淡出与动效前后外推（Temporal Extrapolation）**：
   - 向前/向后各外推 18 帧，100% 消除水印在刚弹出瞬间的胶囊展开、打字机动效及淡出残影。
3. **自适应局部动态亮度掩膜（Adaptive Dynamic Masking）**：
   - 动态识别展开的高亮光晕与字形边缘，结合 Inpainting 进行无缝背景还原。
4. **即开即用**：
   - 支持拖拽多个视频批量处理；
   - 支持双击直接弹出 Windows 原生文件选择对话框；
   - 解决 Windows 中文路径与编码问题。

## 📦 依赖安装

确保已安装 Python 3.8+，然后执行：

```bash
pip install -r requirements.txt
```

## 🚀 使用方法

### 方式一：拖拽处理（推荐）
直接将一个或多个视频文件拖拽到 `一键去水印.bat` 或 `拖拽视频到此处一键去水印.bat` 图标上，工具将自动开始处理。

### 方式二：双击点选
双击运行 `一键去水印.bat`，在弹出的文件选择器中选择需要去水印的视频文件即可。

### 方式三：命令行调用
```bash
python remove_dola_watermark.py "your_video.mp4"
```

处理完成后将在同目录下生成 `[原文件名]_无水印.mp4`。
