import cv2
import numpy as np
import matplotlib.pyplot as plt


def region_of_interest(img, vertices):
    """创建ROI掩码"""
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_lanes(img, lines, color=(0, 255, 0), thickness=10):
    """改进的车道线绘制，区分左右车道线"""
    if lines is None:
        return img

    img_copy = np.copy(img)

    # 分别存储左右车道线
    left_lines = []
    right_lines = []

    # 分离左右车道线
    for line in lines:
        for x1, y1, x2, y2 in line:
            # 计算斜率
            if x2 - x1 == 0:  # 避免除零错误
                continue
            slope = (y2 - y1) / (x2 - x1)

            # 根据斜率判断左右车道线
            if abs(slope) < 0.5:  # 忽略接近水平的线
                continue
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

    # 绘制左车道线
    if left_lines:
        left_points = np.array(left_lines).reshape(-1, 2)
        left_fit = np.polyfit(left_points[:, 1], left_points[:, 0], 1)
        y1 = img.shape[0]
        y2 = int(y1 * 0.6)
        x1 = int(left_fit[0] * y1 + left_fit[1])
        x2 = int(left_fit[0] * y2 + left_fit[1])
        cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)

    # 绘制右车道线
    if right_lines:
        right_points = np.array(right_lines).reshape(-1, 2)
        right_fit = np.polyfit(right_points[:, 1], right_points[:, 0], 1)
        y1 = img.shape[0]
        y2 = int(y1 * 0.6)
        x1 = int(right_fit[0] * y1 + right_fit[1])
        x2 = int(right_fit[0] * y2 + right_fit[1])
        cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)

    return img_copy


def lane_detection_improved(image_path):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    print(f"图像尺寸: {width}x{height}")

    # 2. 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 3. 高斯模糊
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # 4. Canny边缘检测 - 调整参数
    edges = cv2.Canny(blur, 70, 150)

    # 5. 定义ROI - 根据图像调整
    # 默认ROI（梯形区域）
    roi_vertices = np.array([[
        (width * 0.1, height),  # 左下
        (width * 0.45, height * 0.6),  # 左上
        (width * 0.55, height * 0.6),  # 右上
        (width * 0.9, height)  # 右下
    ]], dtype=np.int32)

    # 可视化ROI区域
    roi_visual = np.copy(img)
    cv2.polylines(roi_visual, roi_vertices, True, (255, 0, 0), 3)

    # 6. 应用ROI
    roi_edges = region_of_interest(edges, roi_vertices)

    # 7. 霍夫变换 - 调整参数
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )

    # 8. 绘制车道线
    lane_img = draw_lanes(img, lines)

    # 9. 显示结果
    plt.figure(figsize=(20, 12))

    plt.subplot(2, 3, 1)
    plt.title("1. Original Image")
    plt.imshow(img)

    plt.subplot(2, 3, 2)
    plt.title("2. Gray Image")
    plt.imshow(gray, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title("3. Canny Edges")
    plt.imshow(edges, cmap='gray')

    plt.subplot(2, 3, 4)
    plt.title("4. ROI Region")
    plt.imshow(roi_visual)

    plt.subplot(2, 3, 5)
    plt.title("5. ROI Edges")
    plt.imshow(roi_edges, cmap='gray')

    plt.subplot(2, 3, 6)
    plt.title("6. Lane Detection")
    plt.imshow(lane_img)

    plt.tight_layout()
    plt.show()

    # 保存结果
    cv2.imwrite("output_lane_improved.jpg", cv2.cvtColor(lane_img, cv2.COLOR_RGB2BGR))
    print("结果已保存为 output_lane_improved.jpg")

    # 输出检测到的直线数量
    if lines is not None:
        print(f"检测到 {len(lines)} 条直线")
    else:
        print("未检测到直线")

    return lane_img



# 主程序
if __name__ == "__main__":
    image_path = "test2.jpg"
    result = lane_detection_improved(image_path)