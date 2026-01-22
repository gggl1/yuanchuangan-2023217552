import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # 仅用于读取和保存图像


class ImageProcessor:
    """图像处理器类，实现所有核心算法"""

    def __init__(self, image_path):
        """初始化，读取图像"""
        # 使用PIL读取图像，但后续处理全部自主实现
        self.original_img = np.array(Image.open(image_path))
        self.height, self.width = self.original_img.shape[:2]

        # 将图像转换为灰度图用于滤波
        if len(self.original_img.shape) == 3:
            self.gray_img = self._rgb_to_gray(self.original_img)
        else:
            self.gray_img = self.original_img.copy()

    def _rgb_to_gray(self, rgb_img):
        """将RGB图像转换为灰度图像"""
        # 使用标准公式：Gray = 0.299R + 0.587G + 0.114B
        if len(rgb_img.shape) != 3:
            return rgb_img

        gray = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint8)
        for i in range(rgb_img.shape[0]):
            for j in range(rgb_img.shape[1]):
                gray[i, j] = int(0.299 * rgb_img[i, j, 0] +
                                 0.587 * rgb_img[i, j, 1] +
                                 0.114 * rgb_img[i, j, 2])
        return gray

    def apply_convolution(self, image, kernel):
        """应用卷积核到图像"""
        if len(image.shape) == 3:
            # 如果是彩色图像，分别对每个通道进行卷积
            result = np.zeros_like(image)
            for c in range(3):
                result[:, :, c] = self._single_channel_convolution(image[:, :, c], kernel)
            return result
        else:
            # 灰度图像
            return self._single_channel_convolution(image, kernel)

    def _single_channel_convolution(self, channel, kernel):
        """单通道卷积实现"""
        k_height, k_width = kernel.shape
        pad_h, pad_w = k_height // 2, k_width // 2

        # 为图像添加零填充
        padded = np.zeros((channel.shape[0] + 2 * pad_h, channel.shape[1] + 2 * pad_w))
        padded[pad_h:pad_h + channel.shape[0], pad_w:pad_w + channel.shape[1]] = channel

        # 应用卷积
        result = np.zeros_like(channel, dtype=np.float32)
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                region = padded[i:i + k_height, j:j + k_width]
                result[i, j] = np.sum(region * kernel)

        # 归一化到0-255范围
        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)

    def sobel_filter(self):
        """应用Sobel算子滤波（自主实现）"""
        # Sobel算子
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=np.float32)

        # 分别计算x和y方向的梯度
        grad_x = self._single_channel_convolution(self.gray_img, sobel_x)
        grad_y = self._single_channel_convolution(self.gray_img, sobel_y)

        # 计算梯度幅值
        sobel_result = np.sqrt(grad_x.astype(np.float32) ** 2 + grad_y.astype(np.float32) ** 2)
        sobel_result = np.clip(sobel_result, 0, 255).astype(np.uint8)

        return sobel_result

    def custom_filter(self):
        """应用给定的卷积核滤波"""
        # 给定的卷积核
        custom_kernel = np.array([[1, 0, -1],
                                  [2, 0, -2],
                                  [1, 0, -1]], dtype=np.float32)

        # 应用卷积
        custom_result = self._single_channel_convolution(self.gray_img, custom_kernel)
        return custom_result

    def compute_color_histogram(self, bins=256):
        """计算颜色直方图"""
        if len(self.original_img.shape) != 3:
            # 如果是灰度图，只计算一个通道的直方图
            hist = self._compute_histogram(self.original_img, bins)
            return hist, None, None

        # 分离RGB通道
        r_channel = self.original_img[:, :, 0]
        g_channel = self.original_img[:, :, 1]
        b_channel = self.original_img[:, :, 2]

        # 分别计算各通道的直方图
        r_hist = self._compute_histogram(r_channel, bins)
        g_hist = self._compute_histogram(g_channel, bins)
        b_hist = self._compute_histogram(b_channel, bins)

        return r_hist, g_hist, b_hist

    def _compute_histogram(self, channel, bins):
        """计算单个通道的直方图"""
        hist = np.zeros(bins, dtype=np.int32)
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                pixel_value = channel[i, j]
                if pixel_value < bins:
                    hist[pixel_value] += 1

        # 归一化到0-1范围
        hist_normalized = hist.astype(np.float32) / (channel.shape[0] * channel.shape[1])
        return hist_normalized

    def extract_texture_features(self, method='LBP'):
        """提取纹理特征"""
        if method == 'LBP':
            return self._extract_lbp_features()
        else:
            # 可以扩展其他纹理特征提取方法
            raise ValueError(f"Unsupported method: {method}")

    def _extract_lbp_features(self):
        """使用局部二值模式(LBP)提取纹理特征"""
        # 转换为灰度图（如果还不是灰度图）
        if len(self.original_img.shape) == 3:
            gray = self._rgb_to_gray(self.original_img)
        else:
            gray = self.original_img

        # 初始化LBP图像
        lbp_image = np.zeros((gray.shape[0] - 2, gray.shape[1] - 2), dtype=np.uint8)

        # 计算LBP值
        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                center = gray[i, j]
                binary_pattern = 0

                # 定义8个邻域位置（顺时针）
                neighbors_pos = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                                 (i, j + 1), (i + 1, j + 1), (i + 1, j),
                                 (i + 1, j - 1), (i, j - 1)]

                # 构建二进制模式
                for k, (ni, nj) in enumerate(neighbors_pos):
                    if gray[ni, nj] >= center:
                        binary_pattern |= (1 << (7 - k))

                lbp_image[i - 1, j - 1] = binary_pattern

        # 计算LBP直方图作为纹理特征
        lbp_hist = self._compute_histogram(lbp_image, 256)

        return lbp_image, lbp_hist

    def visualize_results(self, sobel_result, custom_result, hist_data):
        """可视化结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 原始图像
        if len(self.original_img.shape) == 3:
            axes[0, 0].imshow(self.original_img)
        else:
            axes[0, 0].imshow(self.original_img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Sobel滤波结果
        axes[0, 1].imshow(sobel_result, cmap='gray')
        axes[0, 1].set_title('Sobel Filter Result')
        axes[0, 1].axis('off')

        # 自定义卷积核滤波结果
        axes[0, 2].imshow(custom_result, cmap='gray')
        axes[0, 2].set_title('Custom Kernel Filter Result')
        axes[0, 2].axis('off')

        # 颜色直方图
        if len(self.original_img.shape) == 3:
            r_hist, g_hist, b_hist = hist_data
            x = np.arange(256)
            axes[1, 0].plot(x, r_hist, color='red', alpha=0.7, label='Red')
            axes[1, 0].plot(x, g_hist, color='green', alpha=0.7, label='Green')
            axes[1, 0].plot(x, b_hist, color='blue', alpha=0.7, label='Blue')
            axes[1, 0].set_title('Color Histogram (RGB)')
            axes[1, 0].set_xlabel('Pixel Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            gray_hist = hist_data
            x = np.arange(256)
            axes[1, 0].plot(x, gray_hist, color='black')
            axes[1, 0].set_title('Grayscale Histogram')
            axes[1, 0].set_xlabel('Pixel Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)

        # LBP纹理特征图像
        lbp_image, lbp_hist = self._extract_lbp_features()
        axes[1, 1].imshow(lbp_image, cmap='gray')
        axes[1, 1].set_title('LBP Texture Image')
        axes[1, 1].axis('off')

        # LBP直方图
        x = np.arange(256)
        axes[1, 2].plot(x, lbp_hist, color='purple')
        axes[1, 2].set_title('LBP Histogram')
        axes[1, 2].set_xlabel('LBP Pattern')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('experiment_results.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """主函数"""
    # 请替换为您自己拍摄的图像路径
    image_path = "test1.jpg"  # 请替换为实际图像路径

    # 创建图像处理器
    processor = ImageProcessor(image_path)

    # 1. Sobel滤波
    sobel_result = processor.sobel_filter()

    # 2. 自定义卷积核滤波
    custom_result = processor.custom_filter()

    # 3. 计算颜色直方图
    hist_data = processor.compute_color_histogram()

    # 4. 提取纹理特征（LBP）
    lbp_image, lbp_hist = processor.extract_texture_features()

    # 5. 保存纹理特征到npy文件
    np.save('texture_features.npy', lbp_hist)
    print("纹理特征已保存到 texture_features.npy")

    # 6. 保存处理后的图像
    result_img = Image.fromarray(sobel_result)
    result_img.save('sobel_filtered.png')
    print("Sobel滤波结果已保存到 sobel_filtered.png")

    result_img = Image.fromarray(custom_result)
    result_img.save('custom_filtered.png')
    print("自定义卷积核滤波结果已保存到 custom_filtered.png")

    # 7. 可视化结果
    processor.visualize_results(sobel_result, custom_result, hist_data)
    print("实验结果可视化已保存到 experiment_results.png")

    # 打印特征信息
    print(f"\n图像尺寸: {processor.width}x{processor.height}")
    print(f"纹理特征维度: {lbp_hist.shape}")
    print(f"纹理特征示例 (前10个值): {lbp_hist[:10]}")


if __name__ == "__main__":
    main()