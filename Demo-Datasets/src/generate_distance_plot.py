"""
Visualisation utility for the particle distance pipeline. Reads the
per-frame statistics CSV records under results/<video_name>/csv/ and
renders scatter plots of each distance statistic (min, max, mean,
median) and the close-pair ratio against the frame index. Plots are
exported as PNG figures under results/<video_name>/png/.

颗粒间距流水线的可视化工具。读取 results/<视频名>/csv/ 下的逐帧
统计 CSV 文件，绘制各距离指标（最小、最大、均值、中位数）及近距
离对比例关于帧序号的散点图，并以 PNG 格式导出至
results/<视频名>/png/ 目录。
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt

# ── Project paths / 项目路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def generate_plot(video_name):
    """
    Generate scatter plots of distance statistics versus frame index
    from the CSV records associated with a given video name. Expects
    results/<video_name>/csv/{distance_stats,close_pairs_ratio}.csv
    as inputs, and writes PNG figures to results/<video_name>/png/.

    根据指定视频名对应的 CSV 文件，绘制各距离统计指标关于帧序号的
    散点图。输入路径为 results/<视频名>/csv/，输出路径为
    results/<视频名>/png/。
    """
    output_root = RESULTS_DIR / video_name
    stats_csv = output_root / "csv" / "distance_stats.csv"
    close_pairs_csv = output_root / "csv" / "close_pairs_ratio.csv"

    if not stats_csv.exists():
        print(f"文件不存在 / file not found: {stats_csv}")
        return

    png_dir = output_root / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    max_distances = []
    min_distances = []
    mean_distances = []
    median_distances = []

    with open(stats_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 5:
                continue

            frame = int(row[0])
            try:
                max_dist = float(row[1]) if row[1] != 'N/A' else None
                min_dist = float(row[2]) if row[2] != 'N/A' else None
                mean_dist = float(row[3]) if row[3] != 'N/A' else None
                median_dist = float(row[4]) if row[4] != 'N/A' else None

                frames.append(frame)
                max_distances.append(max_dist)
                min_distances.append(min_dist)
                mean_distances.append(mean_dist)
                median_distances.append(median_dist)
            except ValueError:
                continue

    # Filter out frames with missing entries / 过滤缺失值
    valid_frames = []
    valid_max, valid_min, valid_mean, valid_median = [], [], [], []
    for i in range(len(frames)):
        if max_distances[i] is not None:
            valid_frames.append(frames[i])
            valid_max.append(max_distances[i])
            valid_min.append(min_distances[i])
            valid_mean.append(mean_distances[i])
            valid_median.append(median_distances[i])

    if not valid_frames:
        print("无有效数据 / no valid data")
        return

    # Render one scatter plot per statistic, distinguished by colour
    # 为每项统计量分别绘制散点图，并以颜色加以区分
    plot_specs = [
        ('max',    valid_max,    'red',    'Max Distance Over Frames'),
        ('min',    valid_min,    'blue',   'Min Distance Over Frames'),
        ('mean',   valid_mean,   'green',  'Mean Distance Over Frames'),
        ('median', valid_median, 'purple', 'Median Distance Over Frames'),
    ]

    for tag, values, colour, title in plot_specs:
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_frames, values, c=colour, alpha=0.6)
        plt.title(title)
        plt.xlabel('Frame')
        plt.ylabel('Distance (um)')
        plt.grid(True, alpha=0.3)
        out_path = png_dir / f"{tag}_distance.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"saved: {out_path}")

    # Close-pair ratio plot / 近距离对比例图
    if close_pairs_csv.exists():
        ratio_frames, ratio_values = [], []
        with open(close_pairs_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    ratio_frames.append(int(row[0]))
                    ratio_values.append(float(row[1]))
                except ValueError:
                    continue

        if ratio_frames:
            plt.figure(figsize=(10, 6))
            plt.scatter(ratio_frames, ratio_values, c='orange', alpha=0.6)
            plt.title('Close Pairs Ratio (<3 um) Over Frames')
            plt.xlabel('Frame')
            plt.ylabel('Close Pairs Ratio (%)')
            plt.grid(True, alpha=0.3)
            out_path = png_dir / "close_pairs_ratio.png"
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"saved: {out_path}")
    else:
        print(f"文件不存在 / file not found: {close_pairs_csv}")


if __name__ == "__main__":
    # Experimental group (large particles) and control group
    # 实验组（大颗粒）与对照组
    generate_plot("大颗粒识别")
    generate_plot("对照")
