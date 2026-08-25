# AI Video Watermark Remover (AI 视频水印全自动消除工具箱)

基于计算机视觉开发的自动化无缝消除 AI 生成视频水印工具，支持 **Dola AI** 与 **豆包 AI (Doubao)** 的动态全画幅水印与交替跳动水印。

---

## ✨ 核心特性

1. **多模态时序轨迹追踪（Multi-Modal Trajectory Tracker）**：
   - 结合高精度灰度互相关与 Sobel/Laplacian 边缘梯度互相关，无论在浅色/纯白天幕还是深色/黑衣复杂背景下，均能 100% 稳定捕捉水印；
   - 自动识别水印的周期性位置切换（左上/右下交替）与匀速微漂移运动。
2. **动效拟合与时序前后外推（Polynomial Fitting & Temporal Extrapolation）**：
   - 采用一阶多项式方程拟合时序运动轨迹，消除逐帧抖动；
   - 向前/向后各外推 18 帧，彻底消除水印弹出瞬间的展开动效、打字机擦除及淡出残影。
3. **复合掩膜与图像无缝修复（Inpainting Engine）**：
   - 融合文字字形、外轮廓阴影与局部高频差异掩膜；
   - 运用 Navier-Stokes / Telea 图像修复算法，还原自然背景。
4. **原音高保真自动混流（Lossless Audio Remuxing）**：
   - 自动提取并无损重封装原视频高清音频轨道（AAC/MP3 等），确保音画完美同步。
5. **即开即用与极简操作**：
   - 支持拖拽多个视频批量一键处理；
   - 支持双击直接弹出 Windows 原生文件选择对话框；
   - 完美兼容 Windows 中文路径与各类文件名。

---

## 📦 依赖安装

确保已安装 Python 3.8+，然后执行：

```bash
pip install -r requirements.txt
```

---

## 🚀 使用方法

### 豆包 AI 视频水印去除 (`doubao.mp4` / 豆包生成视频)
- **方式一（推荐）**：直接拖拽视频文件到 `一键去豆包水印.bat` 或 `拖拽豆包视频到此处一键去水印.bat`。
- **方式二**：双击运行 `一键去豆包水印.bat`，在弹出的文件选择器中选择视频。
- **方式三（命令行）**：
  ```bash
  python remove_doubao_watermark.py "your_doubao_video.mp4"
  ```

### Dola AI 视频水印去除 (`Dola AI` 动态水印)
- **方式一（推荐）**：直接拖拽视频文件到 `一键去水印.bat` 或 `拖拽视频到此处一键去水印.bat`。
- **方式二**：双击运行 `一键去水印.bat`。
- **方式三（命令行）**：
  ```bash
  python remove_dola_watermark.py "your_dola_video.mp4"
  ```

处理完成后将在同目录下生成 `[原文件名]_无水印.mp4`。
