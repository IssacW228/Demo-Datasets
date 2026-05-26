"""
Control-group particle detection and inter-particle distance
measurement pipeline. Black spherical particles are isolated by direct
inverse grayscale thresholding followed by morphological refinement and
external contour extraction. Pairwise shortest distances and aggregation
statistics are computed per frame and persisted as CSV records.

对照组颗粒识别与颗粒间距测量流水线。通过灰度图直接反向阈值分割并
经形态学精修后，采用外部轮廓提取分离黑色球形颗粒，逐帧计算颗粒对
最短距离与聚集统计量并写入 CSV 文件。
"""

import cv2
import numpy as np
import csv
import os
from pathlib import Path

# ── Project paths / 项目路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Calibration constants / 标定常量
PIXELS_PER_MICRON = 6.2
CLOSE_THRESHOLD_UM = 3.0
CLOSE_THRESHOLD_PX = CLOSE_THRESHOLD_UM * PIXELS_PER_MICRON


def calculate_contour_distance(contour1, contour2):
    """
    Compute the shortest Euclidean boundary distance between two contours
    via adaptive sub-sampling with n_s = max(10, min(50, |C|/10)).

    通过自适应子采样（采样点数 n_s = max(10, min(50, |C|/10))）计算
    两条轮廓之间的最短欧氏边界距离。
    """
    min_distance = float('inf')
    closest_point1 = None
    closest_point2 = None

    sample_points1 = max(10, min(50, len(contour1) // 10))
    sample_points2 = max(10, min(50, len(contour2) // 10))

    points1 = []
    step1 = max(1, len(contour1) // sample_points1)
    for i in range(0, len(contour1), step1):
        points1.append(contour1[i][0])

    points2 = []
    step2 = max(1, len(contour2) // sample_points2)
    for i in range(0, len(contour2), step2):
        points2.append(contour2[i][0])

    for p1 in points1:
        for p2 in points2:
            distance = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_point1 = p1
                closest_point2 = p2

    return min_distance, closest_point1, closest_point2


def process_frame(frame, threshold=127, min_area=20, max_area=1000):
    """
    Per-frame processing pipeline for the control group: inverse binary
    thresholding on the grayscale projection, morphological closing and
    opening with a 3×3 kernel, external contour extraction with an area
    filter, and pairwise shortest-distance estimation.

    对照组单帧处理流水线：在灰度图上执行反向二值阈值分割，采用 3×3
    结构元依次进行形态学闭运算与开运算，提取外部轮廓并施加面积过滤，
    最后估计颗粒对最短距离。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Inverse threshold: dark particles → foreground pixels
    # 反向阈值：深色颗粒映射为前景像素
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Morphological refinement: closing fills internal gaps; opening
    # removes isolated noise.
    # 形态学精修：闭运算填补内部空洞，开运算去除孤立噪声。
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    black_circle_contours = [cnt for cnt in contours
                             if min_area < cv2.contourArea(cnt) < max_area]

    result = frame.copy()
    cv2.drawContours(result, black_circle_contours, -1, (0, 255, 0), 2)

    min_distance = float('inf')
    closest_pair = None
    closest_points = None
    all_distances = []

    if len(black_circle_contours) >= 2:
        for i in range(len(black_circle_contours)):
            for j in range(i + 1, len(black_circle_contours)):
                distance, point1, point2 = calculate_contour_distance(
                    black_circle_contours[i], black_circle_contours[j]
                )
                all_distances.append(distance)
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (i, j)
                    closest_points = (point1, point2)

        if closest_pair and closest_points:
            cv2.drawContours(result, [black_circle_contours[closest_pair[0]]],
                             -1, (0, 0, 255), 3)
            cv2.drawContours(result, [black_circle_contours[closest_pair[1]]],
                             -1, (0, 0, 255), 3)
            point1, point2 = closest_points
            cv2.line(result, tuple(point1.astype(int)),
                     tuple(point2.astype(int)), (255, 0, 0), 2)
            mid_x = int((point1[0] + point2[0]) / 2)
            mid_y = int((point1[1] + point2[1]) / 2)
            cv2.putText(result, f"Distance: {min_distance / PIXELS_PER_MICRON:.2f} um",
                        (mid_x - 50, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(result, f"Circles Detected: {len(black_circle_contours)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    max_distance = float('-inf')
    mean_distance = 0
    median_distance = 0

    if all_distances:
        max_distance = max(all_distances)
        mean_distance = np.mean(all_distances)
        median_distance = np.median(all_distances)

    # Close-pair ratio: denominator restricted to pairs with distance ≤ median,
    # excluding far-apart pairs that carry no aggregation information.
    # 近距离对比例：分母仅包含距离 ≤ 中位数的颗粒对，排除远距离无聚集
    # 意义的配对。
    close_pairs_ratio = 0
    if all_distances:
        close_pairs_count = sum(1 for d in all_distances if d < CLOSE_THRESHOLD_PX)
        relevant_pairs = [d for d in all_distances if d <= median_distance]
        close_pairs_ratio = (close_pairs_count / len(relevant_pairs) * 100
                             if relevant_pairs else 0)

    if len(black_circle_contours) >= 2:
        cv2.putText(result, f"Min Distance: {min_distance / PIXELS_PER_MICRON:.2f} um",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return (result, gray, binary, min_distance, closest_pair, closest_points,
            all_distances, max_distance, mean_distance, median_distance,
            close_pairs_ratio)


def main(video_filename):
    """
    Driver: decode the control-group video frame-by-frame, apply the
    per-frame pipeline, and persist per-frame minimum distance, summary
    statistics, and close-pair ratio under results/<video_name>/csv/.

    主程序：逐帧解码对照组视频，应用单帧流水线，并将逐帧最短距离、
    统计指标汇总及近距离对比例写入 results/<视频名>/csv/ 目录。

    Parameters:
        video_filename : str — filename inside DATA_DIR (e.g. "对照.avi")
    """
    video_path = DATA_DIR / video_filename
    if not video_path.exists():
        print(f"视频文件不存在 / video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频文件 / Failed to open video file: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_name = video_path.stem
    output_root = RESULTS_DIR / video_name
    csv_dir = output_root / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    min_distance_csv = csv_dir / "min_distance.csv"
    stats_csv = csv_dir / "distance_stats.csv"
    close_pairs_csv = csv_dir / "close_pairs_ratio.csv"

    with open(min_distance_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Min_Distance(um)'])

    with open(stats_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Max_Distance(um)', 'Min_Distance(um)',
                         'Mean_Distance(um)', 'Median_Distance(um)'])

    with open(close_pairs_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Close_Pairs_Ratio(%, deno=median)'])

    # Binary video output / Binary 视频输出
    binary_video_path = f"{video_name}_binary.avi"
    binary_out = cv2.VideoWriter(binary_video_path, 
                                 cv2.VideoWriter_fourcc(*'XVID'), 
                                 original_fps, (width, height), isColor=False)

    print(f"原始视频帧率 / Native frame rate: {original_fps:.1f} fps")

    # Playback rate decoupled from native rate for interactive inspection
    # 播放速率与原生帧率解耦，便于交互式观察
    target_fps = 2
    wait_time = int(1000 / target_fps)

    paused = False
    frame_count = 0

    # Default hyper-parameters / 默认超参数
    threshold = 127
    min_area = 20
    max_area = 1000

    print("开始处理视频 / Processing started")
    print(f"输入 / input:   {video_path}")
    print(f"输出 / output:  {output_root}")
    print(f"播放速度 / Playback rate: {target_fps} fps")
    print("控制说明 / Key bindings:")
    print("  q          : 退出 / quit")
    print("  space      : 暂停/继续 / pause toggle")
    print("  + / -      : 阈值 ±10 / threshold ±10")
    print("  6 / 7      : 最小面积 ±10 / min area ±10")
    print("  a / s      : 最大面积 ±100 / max area ±100")

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            (result, gray, binary, min_distance, closest_pair, closest_points,
             all_distances, max_distance, mean_distance, median_distance,
             close_pairs_ratio) = process_frame(
                frame, threshold=threshold,
                min_area=min_area, max_area=max_area,
            )

            with open(min_distance_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if min_distance != float('inf'):
                    writer.writerow([frame_count,
                                     round(min_distance / PIXELS_PER_MICRON, 4)])
                else:
                    writer.writerow([frame_count, 'N/A'])

            with open(stats_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    frame_count,
                    round(max_distance / PIXELS_PER_MICRON, 4) if max_distance != float('-inf') else 'N/A',
                    round(min_distance / PIXELS_PER_MICRON, 4) if min_distance != float('inf') else 'N/A',
                    round(mean_distance / PIXELS_PER_MICRON, 4) if mean_distance != 0 else 'N/A',
                    round(median_distance / PIXELS_PER_MICRON, 4) if median_distance != 0 else 'N/A',
                ])

            with open(close_pairs_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([frame_count, round(close_pairs_ratio, 4)])

            # Write Binary video / 写入 Binary 视频
            binary_out.write(binary)

            cv2.imshow("Original", frame)
            cv2.imshow("Gray", gray)
            cv2.imshow("Binary", binary)
            cv2.imshow("Result", result)

        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"{'已暂停 / paused' if paused else '已继续 / resumed'}")
        elif key == ord('+'):
            threshold = min(255, threshold + 10)
            print(f"threshold = {threshold}")
        elif key == ord('-'):
            threshold = max(0, threshold - 10)
            print(f"threshold = {threshold}")
        elif key == ord('6'):
            min_area = min(500, min_area + 10)
            print(f"min area = {min_area}")
        elif key == ord('7'):
            min_area = max(5, min_area - 10)
            print(f"min area = {min_area}")
        elif key == ord('a'):
            max_area = min(5000, max_area + 100)
            print(f"max area = {max_area}")
        elif key == ord('s'):
            max_area = max(100, max_area - 100)
            print(f"max area = {max_area}")

    binary_out.release()
    cap.release()
    cv2.destroyAllWindows()

    print(f"处理完成 / Processing complete: {frame_count} frames")
    print(f"逐帧最短距离 / per-frame min distance: {min_distance_csv}")
    print(f"距离统计汇总 / distance statistics:    {stats_csv}")
    print(f"近距离对比例 / close-pair ratio:       {close_pairs_csv}")
    print(f"Binary 视频 / Binary video:            {binary_video_path}")


if __name__ == "__main__":
    main("对照.avi")
