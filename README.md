# Micro-particle Detection & Inter-particle Distance Analysis
# 微颗粒检测与颗粒间距分析

A classical computer vision pipeline for detecting spherical micro-particles in microscopy video sequences and quantifying inter-particle distances over time. Two independent detection strategies are implemented: a **red-shadow attenuation pipeline** for the experimental group (large particles with illumination artefacts) and a **direct inverse-threshold pipeline** for the control group (black particles on a uniform background).

基于经典计算机视觉方法的显微视频颗粒检测与颗粒间距时序分析流水线。针对两类实验条件分别实现了独立的检测策略：面向实验组（存在光照红色伪影的大颗粒）的**红色阴影衰减流水线**，以及面向对照组（均匀背景下的黑色颗粒）的**直接反向阈值流水线**。

---

## Table of Contents / 目录

- [Project Structure / 项目结构](#project-structure--项目结构)
- [Pipeline Overview / 算法流水线概览](#pipeline-overview--算法流水线概览)
- [Requirements / 依赖环境](#requirements--依赖环境)
- [Usage / 使用说明](#usage--使用说明)
- [Output Files / 输出文件说明](#output-files--输出文件说明)
- [Key Parameters / 关键参数](#key-parameters--关键参数)
- [Documentation / 文档](#documentation--文档)

---

## Project Structure / 项目结构

```
med_cv_project/
│
├── src/                              # Source code / 源代码
│   ├── red_shadow_removal.py         # Experimental group pipeline (large particles)
│   │                                 # 实验组流水线（大颗粒，含红色阴影衰减）
│   ├── black_circle_distance_analysis.py
│   │                                 # Control group pipeline (black particles)
│   │                                 # 对照组流水线（黑色颗粒，直接阈值分割）
│   └── generate_distance_plot.py     # Visualisation utility (CSV → scatter plots)
│                                     # 可视化工具（CSV → 散点图）
│
├── data/                             # Raw input videos / 原始输入视频
│   ├── 大颗粒识别.avi                 # Experimental group video / 实验组视频
│   └── 对照.avi                      # Control group video / 对照组视频
│
├── results/                          # Auto-generated pipeline outputs
│   │                                 # 自动生成的流水线输出
│   ├── 大颗粒识别/                    # One directory per video / 每个视频一个子目录
│   │   ├── csv/
│   │   │   ├── min_distance.csv      # Per-frame minimum distance
│   │   │   ├── distance_stats.csv    # Per-frame max/min/mean/median
│   │   │   └── close_pairs_ratio.csv # Close-pair ratio (< 3 μm)
│   │   ├── png/
│   │   │   ├── max_distance.png
│   │   │   ├── min_distance.png
│   │   │   ├── mean_distance.png
│   │   │   ├── median_distance.png
│   │   │   └── close_pairs_ratio.png
│   │   ├── annotated.avi             # Full-pipeline visualisation
│   │   └── white_circles.avi         # Extracted particle mask
│   │
│   └── 对照/
│       ├── csv/
│       │   ├── min_distance.csv
│       │   ├── distance_stats.csv
│       │   └── close_pairs_ratio.csv
│       ├── png/
│       │   ├── max_distance.png
│       │   ├── min_distance.png
│       │   ├── mean_distance.png
│       │   ├── median_distance.png
│       │   └── close_pairs_ratio.png
│       └── binary.avi                # Post-morphological binary mask sequence
│                                     # 经形态学精修后的二值掩码序列
│
├── docs/                             # Documentation / 文档
│   ├── Particle_Detection_Algorithm_Specification_Compact.docx
│   │                                 # Algorithm specification (EN) / 算法说明文档
│   └── Pipeline_Diagram.png          # Pipeline diagram / 流水线架构图
│
├── .gitignore
└── README.md
```

> **Note:** `data/` and `results/` contents are **not tracked by Git**. Place raw videos in `data/` before running. Each detection script auto-creates `results/<video_name>/csv/` and writes its annotated video to `results/<video_name>/`. The plotting script auto-creates `results/<video_name>/png/`.
>
> `data/` 与 `results/` 内容不纳入 Git 版本控制。运行前请将原始视频放入 `data/`。检测脚本会自动在 `results/<视频名>/csv/` 下创建 CSV 文件，并把标注视频写入 `results/<视频名>/`。绘图脚本会自动创建 `results/<视频名>/png/`。

---

## Pipeline Overview / 算法流水线概览

![Pipeline Diagram](docs/Pipeline_Diagram.png)

The text diagram below mirrors the figure above for quick reference in any Markdown renderer. Module groups: **M1** red-shadow attenuation, **M2** binarisation & morphological refinement, **M3–M4** contour extraction & distance estimation, **Outputs** persistence.

下方文本流程图与上图对应，便于在任意 Markdown 环境中快速参阅。模块分组：**M1** 红色阴影衰减、**M2** 二值化与形态学精修、**M3–M4** 轮廓提取与距离估计、**Outputs** 持久化输出。

```
Input frame  (BGR microscopy video)
    │
    ▼
┌─ M1 ─────────────────────────────────────────────────┐
│  Red-shadow attenuation                              │
│  HSV mask · hue ∈ [0°,15°]∪[160°,180°] · sat ÷ 3     │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌─ M2 ─────────────────────────────────────────────────┐
│  Binarisation                                        │
│  CLAHE (clip 5.0) + adaptive Gaussian threshold      │
│  (block 51, C = 25) + image inversion                │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Morphological refinement                            │
│  Dilation–erosion · 3×3 kernel                       │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌─ M3 ─────────────────────────────────────────────────┐
│  Hierarchical contour extraction                     │
│  RETR_TREE · area filter [20, 1000] px²              │
│  parent index ≠ −1 (retain enclosed inner holes)     │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌─ M4 ─────────────────────────────────────────────────┐
│  Inter-particle distance estimation                  │
│  Sub-sampled pairwise Euclidean minimum              │
│  n_s = max(10, min(50, |C|/10))                      │
└──────────────────────────────────────────────────────┘
    │
    ├──────────────────┬──────────────────┐
    ▼                  ▼                  ▼
┌─ Output ────┐  ┌─ Output ────┐  ┌─ Output ─────┐
│ Per-frame   │  │ CSV records │  │ Scatter      │
│ statistics  │  │             │  │ plots        │
│             │  │ 3 files     │  │              │
│ min / max / │  │ per video   │  │ Distance vs  │
│ mean /      │  │ sequence    │  │ frame index  │
│ median +    │  │             │  │              │
│ close-pair  │  │             │  │ PNG via      │
│ ratio       │  │             │  │ Matplotlib   │
│ (< 3 μm)    │  │             │  │              │
└─────────────┘  └─────────────┘  └──────────────┘
```

### Experimental Group — `red_shadow_removal.py` / 实验组

| Stage | Operation | 操作 |
|---|---|---|
| 1 | Red-shadow attenuation (HSV, hue ∈ [0°,15°]∪[160°,180°], saturation ÷ 3) | HSV 空间红色阴影衰减 |
| 2 | Grayscale projection + optional CLAHE (clip 5.0, grid 8×8) | 灰度投影 + 可选 CLAHE 增强 |
| 3 | Adaptive Gaussian threshold (block 51, C = 25) + inversion | 自适应高斯阈值 + 图像取反 |
| 4 | Morphological dilation–erosion (3×3 kernel) | 膨胀–腐蚀形态学精修 |
| 5 | RETR_TREE contour extraction, retain inner holes (parent ≠ −1) | 层次轮廓提取，保留内部孔洞 |
| 6 | Pairwise shortest-distance estimation via adaptive sub-sampling | 自适应子采样颗粒对最短距离估计 |

### Control Group — `black_circle_distance_analysis.py` / 对照组

| Stage | Operation | 操作 |
|---|---|---|
| 1 | Grayscale projection | 灰度投影 |
| 2 | Inverse binary threshold (default 127) | 反向二值阈值分割 |
| 3 | Morphological closing + opening (3×3 kernel) | 形态学闭/开运算精修 |
| 4 | RETR_EXTERNAL contour extraction, area filter | 外部轮廓提取，面积过滤 |
| 5 | Pairwise shortest-distance estimation | 颗粒对最短距离估计 |

### Calibration constant / 标定常量

```
PIXELS_PER_MICRON = 6.2 px/μm
CLOSE_THRESHOLD   = 3.0 μm  (= 18.6 px)
```

---

## Requirements / 依赖环境

```bash
pip install opencv-python numpy matplotlib
```

| Package | Version tested | Purpose |
|---|---|---|
| `opencv-python` | ≥ 4.8 | Video I/O, image processing / 视频读写与图像处理 |
| `numpy` | ≥ 1.24 | Numerical computation / 数值计算 |
| `matplotlib` | ≥ 3.7 | Scatter plot generation / 散点图生成 |

> Python ≥ 3.9 recommended / 建议使用 Python ≥ 3.9

---

## Usage / 使用说明

All scripts use `pathlib` with the project root anchored to the script location, so they can be invoked from **any working directory** without path errors.

所有脚本以脚本自身位置为锚点解析路径，可从**任意工作目录**运行而不会路径出错。

### Step 1 — Place input videos / 放置输入视频

Put your videos in `data/`:
```
data/
├── 大颗粒识别.avi
└── 对照.avi
```

### Step 2 — Run detection / 运行检测

**Experimental group (large particles with red shadow):**
```bash
python src/red_shadow_removal.py
```

**Control group (black particles):**
```bash
python src/black_circle_distance_analysis.py
```

Both scripts open **interactive windows** during processing. Key bindings / 键盘控制：

| Key | Action | 说明 |
|---|---|---|
| `q` | Quit / 退出 | |
| `Space` | Pause / resume | 暂停 / 继续 |
| `+` / `-` | Threshold ± 10 | 阈值调整（固定阈值模式） |
| `1` / `2` | Fixed / adaptive threshold | 切换阈值方法（仅实验组） |
| `3` | Toggle mean ↔ gaussian | 切换自适应方法（仅实验组） |
| `6` / `7` | Min area ± 5 / ± 10 | 最小面积调整 |
| `8` / `9` | Block size ± 2 | 自适应阈值块大小（仅实验组） |
| `0` | C value + 1 | 自适应阈值 C 值（仅实验组） |
| `a` / `s` | Max area ± 100 | 最大面积调整 |
| `c` | Toggle CLAHE | 切换 CLAHE（仅实验组） |
| `=` / `_` | CLAHE clip ± 0.5 | CLAHE 限幅阈值（仅实验组） |

### Step 3 — Generate plots / 生成散点图

```bash
python src/generate_distance_plot.py
# Reads CSV records from results/<video_name>/csv/ and writes
# PNG scatter plots to results/<video_name>/png/
# 从 results/<视频名>/csv/ 读取 CSV，输出 PNG 散点图至 results/<视频名>/png/
```

### Quick start (run all) / 一键运行

```bash
python src/red_shadow_removal.py          # Process experimental group
python src/black_circle_distance_analysis.py  # Process control group
python src/generate_distance_plot.py      # Generate all scatter plots
```

---

## Output Files / 输出文件说明

For each video `<video_name>`, the pipeline produces the following under `results/<video_name>/`:

针对每段视频 `<视频名>`，流水线在 `results/<视频名>/` 下生成：

| File | Columns / Content | Description |
|---|---|---|
| `csv/min_distance.csv` | `Frame, Min_Distance(um)` | Per-frame closest particle pair / 逐帧最近颗粒对 |
| `csv/distance_stats.csv` | `Frame, Max, Min, Mean, Median (um)` | Per-frame distance statistics / 逐帧距离统计 |
| `csv/close_pairs_ratio.csv` | `Frame, Ratio(%, deno=median)` | Proportion of pairs with *d* < 3 μm, denominator = pairs with *d* ≤ *d*<sub>median</sub> / 近距离对比例 |
| `png/{max,min,mean,median}_distance.png` | Scatter plot vs frame index | Distance trajectory plots / 距离时序散点图 |
| `png/close_pairs_ratio.png` | Scatter plot vs frame index | Aggregation tendency over time / 聚集趋势时序图 |
| `binary.avi` *(control only)* | XVID-encoded grayscale video | Post-morphological binary mask sequence / 经形态学精修后的二值掩码序列 |
| `annotated.avi` *(experimental only)* | XVID-encoded video | Full pipeline visualisation / 全流水线可视化 |
| `white_circles.avi` *(experimental only)* | XVID-encoded video | Extracted particle mask / 颗粒掩码 |

---

## Key Parameters / 关键参数

All hyper-parameters can be tuned interactively at runtime via keyboard bindings, or set by modifying the defaults in `main()`:

所有超参数均可在运行时通过键盘交互调优，也可直接修改 `main()` 中的默认值：

| Parameter | Default (experimental) | Default (control) | Description |
|---|---|---|---|
| `threshold` | 64 | 127 | Fixed threshold value / 固定阈值 |
| `threshold_method` | `'adaptive'` | `'fixed'` | Thresholding strategy |
| `block_size` | 51 | — | Adaptive threshold neighbourhood / 自适应阈值块大小 |
| `c_value` | 25 | — | Adaptive threshold constant *C* |
| `use_clahe` | `True` | — | Enable CLAHE / 启用 CLAHE |
| `clahe_clip_limit` | 5.0 | — | CLAHE clip limit / 限幅阈值 |
| `min_area` | 20 px² | 20 px² | Minimum contour area / 最小轮廓面积 |
| `max_area` | 1000 px² | 1000 px² | Maximum contour area / 最大轮廓面积 |

---

## Documentation / 文档

- **Algorithm specification (EN):** [`docs/Particle_Detection_Algorithm_Specification_Compact.docx`](docs/Particle_Detection_Algorithm_Specification_Compact.docx)
- **Pipeline diagram:** [`docs/Pipeline_Diagram.png`](docs/Pipeline_Diagram.png)

The specification covers the full mathematical formulation of the pipeline, including the formal definition of the close-pair ratio with median-restricted denominator.

算法说明文档包含流水线的完整数学表述，以及近距离对比例中以中位数限制分母的形式化定义。

---

## Notes for GitHub Hosting / GitHub 托管注意事项

- **Video files (`*.avi`)** are excluded from version control (see `.gitignore`). They typically exceed GitHub's 25 MB soft limit per file. Options for sharing:
  - Use **Git LFS**: `git lfs track "*.avi"` then commit normally
  - Host on Google Drive / OSF / Zenodo and link from this README
- 视频文件不入库，因单文件常超过 GitHub 25 MB 软上限。可选方案：
  - 使用 **Git LFS** 追踪 `*.avi`
  - 上传至 Google Drive / OSF / Zenodo 并在 README 中附链接
- **Generated outputs** (`results/`) are also gitignored. They are reproducible from `src/` + `data/`.
- 生成产物（`results/`）同样不入库，可由 `src/` + `data/` 重新生成。
