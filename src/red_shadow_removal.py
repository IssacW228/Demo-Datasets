"""
Particle detection and inter-particle distance measurement pipeline
for large-particle microscopy video, based on red-shadow attenuation
in HSV space followed by adaptive thresholding and hierarchical contour
extraction. Inner-hole contours within the dark particle regions are
retained as detected cross-sections.

基于 HSV 空间红色阴影衰减、自适应阈值分割与层次轮廓提取的大颗粒
显微视频识别与颗粒间距测量流水线。算法保留位于深色颗粒区域内部
的孔洞轮廓作为颗粒截面。
"""

import cv2
import numpy as np
import csv
import os
from pathlib import Path

# ── Project paths / 项目路径 ──
# Anchor on this file's location so the script can be invoked from
# any working directory without breaking path resolution.
# 以脚本自身位置为锚点，使脚本从任意工作目录运行均能正确解析路径。
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
    via adaptive sub-sampling, where the number of sample points per
    contour is bounded by n_s = max(10, min(50, |C|/10)) to balance
    accuracy and computational cost.

    通过自适应子采样计算两条轮廓之间的最短欧氏边界距离。每条轮廓的
    采样点数取 n_s = max(10, min(50, |C|/10))，以兼顾精度与计算开销。

    Returns:
        min_distance : float — shortest pairwise distance in pixels
        closest_point1, closest_point2 : ndarray — coordinates of the
            point pair realising the minimum
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


def remove_red_shadow(image, method='fade'):
    """
    Suppress red-shadow illumination artefacts in HSV space.
    The red region is identified by combining hue intervals [0°, 15°]
    and [160°, 180°] via bitwise OR, refined through morphological
    closing and opening (5×5 kernel). 'fade' attenuates the saturation
    channel within the mask by a factor of three; 'remove' replaces
    masked pixels with a white background.

    在 HSV 空间中抑制红色阴影伪影。通过按位或运算融合 [0°, 15°] 与
    [160°, 180°] 两个色调区间构造红色掩码，并采用 5×5 结构元执行形
    态学闭/开运算去噪填洞。'fade' 将掩码区域饱和度衰减为原值的三分
    之一；'remove' 将掩码像素替换为白色背景。

    Returns:
        result : ndarray — image with the red shadow suppressed
        red_mask : ndarray — binary mask of the red region
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 30, 30])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    if method == 'remove':
        white_background = np.ones_like(image) * 255
        non_red_mask = cv2.bitwise_not(red_mask)
        result = cv2.bitwise_and(image, image, mask=non_red_mask)
        result = cv2.bitwise_or(result, white_background, mask=red_mask)
    else:
        h, s, v = cv2.split(hsv)
        s[red_mask > 0] = s[red_mask > 0] // 3
        hsv = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result, red_mask


def process_frame(frame, threshold=64, min_area=20, max_area=1000,
                  threshold_method='adaptive', adaptive_method='gaussian',
                  block_size=11, c_value=2, use_clahe=False,
                  clahe_clip_limit=2.0, clahe_grid_size=(8, 8)):
    """
    Per-frame processing pipeline: (i) red-shadow attenuation,
    (ii) grayscale projection with optional CLAHE enhancement,
    (iii) adaptive Gaussian thresholding, (iv) dilation–erosion
    refinement, (v) hierarchical contour extraction with retention of
    enclosed inner holes (parent index ≠ −1), and (vi) pairwise
    shortest-distance estimation between detected particles.

    单帧处理流水线：(i) 红色阴影衰减；(ii) 灰度投影并可选 CLAHE 增
    强；(iii) 自适应高斯阈值；(iv) 膨胀–腐蚀精修；(v) 层次轮廓提取
    并保留具有父节点的内部孔洞轮廓；(vi) 颗粒对最短距离估计。
    """
    processed_frame, red_mask = remove_red_shadow(frame, method='fade')

    gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit,
                                tileGridSize=clahe_grid_size)
        gray = clahe.apply(gray)

    if threshold_method == 'fixed':
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    else:
        adaptive_flag = (cv2.ADAPTIVE_THRESH_MEAN_C if adaptive_method == 'mean'
                         else cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        thresh = cv2.adaptiveThreshold(gray, 255, adaptive_flag,
                                       cv2.THRESH_BINARY, block_size, c_value)

    # Invert so that dark particle interiors become foreground pixels
    # 取反，使深色颗粒区域对应前景像素
    thresh = cv2.bitwise_not(thresh)

    # Dilation–erosion cycle: bridge fragmented boundaries, suppress noise
    # 膨胀–腐蚀循环：连接断裂边界并抑制噪声
    kernel = np.ones((3, 3), np.uint8)
    dilated_thresh = cv2.dilate(thresh, kernel, iterations=1)
    eroded_thresh = cv2.erode(dilated_thresh, kernel, iterations=1)
    thresh = eroded_thresh

    # Retrieve full contour hierarchy for parent–child analysis
    # 获取完整轮廓层次以进行父子关系分析
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Retain contours satisfying both area constraint and parent index ≠ −1,
    # i.e. inner holes enclosed by an outer dark region — these correspond
    # to genuine particle cross-sections.
    # 保留同时满足面积约束与父节点索引非负的轮廓，即被外层深色区域
    # 包围的内部孔洞，对应真实的颗粒截面。
    white_circle_contours = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if min_area < area < max_area and hierarchy[0][i][3] != -1:
            white_circle_contours.append(contour)

    result = frame.copy()
    cv2.drawContours(result, white_circle_contours, -1, (0, 255, 0), 2)

    white_circles = np.zeros_like(thresh)
    cv2.drawContours(white_circles, white_circle_contours, -1, 255, -1)

    min_distance = float('inf')
    closest_pair = None
    closest_points = None
    all_distances = []

    if len(white_circle_contours) >= 2:
        for i in range(len(white_circle_contours)):
            for j in range(i + 1, len(white_circle_contours)):
                distance, point1, point2 = calculate_contour_distance(
                    white_circle_contours[i], white_circle_contours[j]
                )
                all_distances.append(distance)
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (i, j)
                    closest_points = (point1, point2)

        if closest_pair and closest_points:
            cv2.drawContours(result, [white_circle_contours[closest_pair[0]]],
                             -1, (0, 0, 255), 3)
            cv2.drawContours(result, [white_circle_contours[closest_pair[1]]],
                             -1, (0, 0, 255), 3)
            point1, point2 = closest_points
            cv2.line(result, tuple(point1.astype(int)),
                     tuple(point2.astype(int)), (255, 0, 0), 2)
            mid_x = int((point1[0] + point2[0]) / 2)
            mid_y = int((point1[1] + point2[1]) / 2)
            cv2.putText(result, f"Distance: {min_distance / PIXELS_PER_MICRON:.2f} um",
                        (mid_x - 50, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(result, f"Holes Detected: {len(white_circle_contours)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(result, f"Threshold: {threshold}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

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

    if len(white_circle_contours) >= 2:
        cv2.putText(result, f"Min Distance: {min_distance / PIXELS_PER_MICRON:.2f} um",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return (result, gray, thresh, eroded_thresh, processed_frame, red_mask,
            white_circles, min_distance, closest_pair, closest_points,
            all_distances, max_distance, mean_distance, median_distance,
            close_pairs_ratio)


def main(video_filename):
    """
    Driver: decode the input video frame-by-frame via OpenCV
    VideoCapture, apply the per-frame pipeline, and persist per-frame
    minimum distance, summary statistics, and close-pair ratio as
    separate CSV records under results/<video_name>/csv/. Keyboard
    bindings expose all hyper-parameters for interactive grid search.

    主程序：通过 OpenCV VideoCapture 接口逐帧解码输入视频，应用单帧
    流水线，并将逐帧最短距离、统计指标汇总及近距离对比例分别写入
    results/<视频名>/csv/ 目录下的 CSV 文件。键盘绑定开放全部超参数
    以支持交互式网格搜索。

    Parameters:
        video_filename : str — filename inside DATA_DIR (e.g. "大颗粒识别.avi")
    """
    # Resolve input path under DATA_DIR / 解析 DATA_DIR 下的输入路径
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

    # Output layout under RESULTS_DIR / 输出目录布局：
    #   results/<video_name>/
    #     ├── csv/
    #     │   ├── min_distance.csv
    #     │   ├── distance_stats.csv
    #     │   └── close_pairs_ratio.csv
    #     ├── annotated.avi          (full-pipeline visualisation)
    #     └── white_circles.avi      (binary particle mask)
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

    frames = []
    max_distances = []
    min_distances = []
    mean_distances = []
    median_distances = []

    print(f"原始视频帧率 / Native frame rate: {original_fps:.1f} fps")

    # Playback rate decoupled from native rate for interactive inspection
    # 播放速率与原生帧率解耦，便于交互式观察
    target_fps = 2
    wait_time = int(1000 / target_fps)

    annotated_video = output_root / "annotated.avi"
    white_circles_video = output_root / "white_circles.avi"

    out = cv2.VideoWriter(str(annotated_video),
                          cv2.VideoWriter_fourcc(*'XVID'),
                          original_fps, (width, height))

    white_circles_out = cv2.VideoWriter(str(white_circles_video),
                                        cv2.VideoWriter_fourcc(*'XVID'),
                                        original_fps, (width, height))

    paused = False
    frame_count = 0

    # Default hyper-parameters (tuned via interactive grid search)
    # 默认超参数（经交互式网格搜索调优）
    threshold = 64
    min_area = 20
    max_area = 1000
    threshold_method = 'adaptive'
    adaptive_method = 'gaussian'
    block_size = 51
    c_value = 25
    use_clahe = True
    clahe_clip_limit = 5.0
    clahe_grid_size = (8, 8)

    print("开始处理视频 / Processing started")
    print(f"输入 / input:   {video_path}")
    print(f"输出 / output:  {output_root}")
    print(f"播放速度 / Playback rate: {target_fps} fps")
    print("控制说明 / Key bindings:")
    print("  q          : 退出 / quit")
    print("  space      : 暂停/继续 / pause toggle")
    print("  + / -      : 阈值 +10 / -10 (固定阈值模式) / threshold ±10")
    print("  1 / 2      : 固定 / 自适应阈值 / fixed | adaptive threshold")
    print("  3          : 切换自适应方法 / toggle adaptive method (mean ↔ gaussian)")
    print("  6 / 7      : 最小面积 +5 / -5 / min area ±5")
    print("  8 / 9      : 块大小 +2 / -2 (奇数) / block size ±2 (odd)")
    print("  0          : C 值 +1 / C value +1")
    print("  a / s      : 最大面积 +100 / -100 / max area ±100")
    print("  c          : 切换 CLAHE / toggle CLAHE")
    print("  = / _      : CLAHE 限幅阈值 +0.5 / -0.5 / CLAHE clip limit ±0.5")
    print(f"当前阈值 / threshold = {threshold}, 方法 / method = {threshold_method} ({adaptive_method})")
    print(f"块大小 / block size = {block_size}, C = {c_value}, "
          f"面积区间 / area = [{min_area}, {max_area}]")
    print(f"CLAHE: {'on' if use_clahe else 'off'}, "
          f"限幅 / clip = {clahe_clip_limit:.1f}, 网格 / grid = {clahe_grid_size}")

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            (result, gray, thresh, eroded_thresh, processed_frame, red_mask,
             white_circles, min_distance, closest_pair, closest_points,
             all_distances, max_distance, mean_distance, median_distance,
             close_pairs_ratio) = process_frame(
                frame,
                threshold=threshold,
                min_area=min_area,
                max_area=max_area,
                threshold_method=threshold_method,
                adaptive_method=adaptive_method,
                block_size=block_size,
                c_value=c_value,
                use_clahe=use_clahe,
                clahe_clip_limit=clahe_clip_limit,
                clahe_grid_size=clahe_grid_size,
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

            frames.append(frame_count)
            max_distances.append(max_distance if max_distance != float('-inf') else None)
            min_distances.append(min_distance if min_distance != float('inf') else None)
            mean_distances.append(mean_distance if mean_distance != 0 else None)
            median_distances.append(median_distance if median_distance != 0 else None)

            cv2.imshow("Original", frame)
            cv2.imshow("Processed (Red Shadow Removed)", processed_frame)
            cv2.imshow("Red Mask", red_mask)
            cv2.imshow("Gray", gray)
            cv2.imshow("Threshold", thresh)
            cv2.imshow("Eroded", eroded_thresh)
            cv2.imshow("White Circles", white_circles)
            cv2.imshow("Result", result)

            out.write(result)
            white_circles_bgr = cv2.cvtColor(white_circles, cv2.COLOR_GRAY2BGR)
            white_circles_out.write(white_circles_bgr)

        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"{'已暂停 / paused' if paused else '已继续 / resumed'}")
        elif key == ord('+'):
            threshold += 10
            print(f"threshold = {threshold}")
        elif key == ord('-'):
            threshold = max(0, threshold - 10)
            print(f"threshold = {threshold}")
        elif key == ord('1'):
            threshold_method = 'fixed'
            print("threshold method = fixed")
        elif key == ord('2'):
            threshold_method = 'adaptive'
            print("threshold method = adaptive")
        elif key == ord('3'):
            adaptive_method = 'mean' if adaptive_method == 'gaussian' else 'gaussian'
            print(f"adaptive method = {adaptive_method}")
        elif key == ord('6'):
            min_area = min(100, min_area + 5)
            print(f"min area = {min_area}")
        elif key == ord('7'):
            min_area = max(5, min_area - 5)
            print(f"min area = {min_area}")
        elif key == ord('8'):
            block_size = min(71, block_size + 2)
            print(f"block size = {block_size}")
        elif key == ord('9'):
            block_size = max(3, block_size - 2)
            print(f"block size = {block_size}")
        elif key == ord('0'):
            c_value = min(30, c_value + 1)
            print(f"C = {c_value}")
        elif key == ord('a'):
            max_area = min(5000, max_area + 100)
            print(f"max area = {max_area}")
        elif key == ord('s'):
            max_area = max(100, max_area - 100)
            print(f"max area = {max_area}")
        elif key == ord('c'):
            use_clahe = not use_clahe
            print(f"CLAHE = {'on' if use_clahe else 'off'}")
        elif key == ord('='):
            clahe_clip_limit = min(10.0, clahe_clip_limit + 0.5)
            print(f"CLAHE clip = {clahe_clip_limit:.1f}")
        elif key == ord('_'):
            clahe_clip_limit = max(0.5, clahe_clip_limit - 0.5)
            print(f"CLAHE clip = {clahe_clip_limit:.1f}")

    cap.release()
    out.release()
    white_circles_out.release()
    cv2.destroyAllWindows()

    print(f"处理完成 / Processing complete: {frame_count} frames")
    print(f"逐帧最短距离 / per-frame min distance: {min_distance_csv}")
    print(f"距离统计汇总 / distance statistics:    {stats_csv}")
    print(f"近距离对比例 / close-pair ratio:       {close_pairs_csv}")
    print(f"标注视频 / annotated video:            {annotated_video}")
    print(f"颗粒掩码 / particle mask video:        {white_circles_video}")


if __name__ == "__main__":
    main("大颗粒识别.avi")
