import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from train_3 import DigitRecognizer


class StudentIDRecognizer:
    """学号识别器"""

    def __init__(self, model_path='models/digit_model.pth'):
        self.digit_recognizer = DigitRecognizer(model_path)
        if not self.digit_recognizer.load_model():
            print("请先训练模型！")
            exit(1)

    def preprocess_image(self, image_path):
        """预处理学号图片"""
        # 读取图片
        if isinstance(image_path, str):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = image_path

        if image is None:
            raise ValueError("无法读取图像")

        # 二值化
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 降噪
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def segment_digits(self, binary_image):
        """分割数字"""
        # 寻找轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digit_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # 过滤太小的区域
            if w > 10 and h > 20:
                digit_boxes.append({
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'contour': contour
                })

        # 按x坐标排序（从左到右）
        digit_boxes.sort(key=lambda b: b['x'])

        return digit_boxes

    def extract_digit_images(self, binary_image, digit_boxes):
        """提取单个数字图像"""
        digit_images = []

        for box in digit_boxes:
            x, y, w, h = box['x'], box['y'], box['w'], box['h']

            # 提取数字区域
            digit = binary_image[y:y + h, x:x + w]

            # 调整大小到MNIST格式 (28x28)
            digit = self.resize_to_mnist(digit)

            # 转换为张量
            tensor = self.digit_recognizer.transform(Image.fromarray(digit))

            digit_images.append({
                'tensor': tensor,
                'box': box,
                'original': digit
            })

        return digit_images

    def resize_to_mnist(self, digit):
        """调整到MNIST格式"""
        # 确保是28x28，保持数字在中心
        size = 28
        h, w = digit.shape

        # 计算缩放比例
        scale = min(size / h, size / w) * 0.8

        # 缩放
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(digit, (new_w, new_h))

        # 创建28x28画布
        canvas = np.zeros((size, size), dtype=np.uint8)

        # 将数字放在中心
        x_offset = (size - new_w) // 2
        y_offset = (size - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas

    def recognize_digits(self, digit_images):
        """识别数字"""
        predictions = []

        with torch.no_grad():
            for digit_info in digit_images:
                tensor = digit_info['tensor'].unsqueeze(0)  # 添加batch维度
                tensor = tensor.to(self.digit_recognizer.device)

                output = self.digit_recognizer.model(tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                predictions.append({
                    'digit': predicted.item(),
                    'confidence': confidence.item(),
                    'box': digit_info['box'],
                    'image': digit_info['original']
                })

        return predictions

    def visualize_results(self, original_image, predictions):
        """可视化识别结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 原始图像
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original Student ID Image')
        axes[0, 0].axis('off')

        # 预处理后的图像
        binary = self.preprocess_image(original_image)
        axes[0, 1].imshow(binary, cmap='gray')
        axes[0, 1].set_title('Preprocessed Image')
        axes[0, 1].axis('off')

        # 数字分割结果
        digit_boxes = self.segment_digits(binary)
        color_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

        for box in digit_boxes:
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        axes[1, 0].imshow(color_image)
        axes[1, 0].set_title('Digit Segmentation')
        axes[1, 0].axis('off')

        # 识别结果显示
        student_id = ''.join([str(p['digit']) for p in predictions])
        confidence = np.mean([p['confidence'] for p in predictions])

        axes[1, 1].text(0.1, 0.5, f'识别结果: {student_id}\n\n'
                                  f'平均置信度: {confidence:.2%}\n\n'
                                  f'数字个数: {len(predictions)}',
                        fontsize=14, verticalalignment='center')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Recognition Result')

        plt.tight_layout()
        plt.savefig('student_id_recognition_result.png', dpi=150)
        plt.show()

        return student_id

    def recognize_student_id(self, image_path):
        """识别学号主函数"""
        print("开始识别学号...")

        # 读取图像
        if isinstance(image_path, str):
            original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            original_image = image_path

        # 1. 预处理
        binary_image = self.preprocess_image(original_image)

        # 2. 分割数字
        digit_boxes = self.segment_digits(binary_image)
        print(f"检测到 {len(digit_boxes)} 个数字")

        # 3. 提取数字图像
        digit_images = self.extract_digit_images(binary_image, digit_boxes)

        # 4. 识别数字
        predictions = self.recognize_digits(digit_images)

        # 5. 显示结果
        student_id = self.visualize_results(original_image, predictions)

        # 6. 打印详细结果
        print("\n数字识别结果:")
        print("-" * 40)
        for i, pred in enumerate(predictions):
            print(f"数字 {i + 1}: {pred['digit']} (置信度: {pred['confidence']:.2%})")
        print("-" * 40)
        print(f"识别出的学号: {student_id}")

        # 保存结果到文件
        with open('recognition_result.txt', 'w', encoding='utf-8') as f:
            f.write("学号识别结果\n")
            f.write("=" * 30 + "\n")
            f.write(f"识别时间: {np.datetime64('now')}\n")
            f.write(f"检测到的数字个数: {len(predictions)}\n")
            f.write(f"识别出的学号: {student_id}\n\n")
            f.write("详细识别结果:\n")
            for i, pred in enumerate(predictions):
                f.write(f"  数字 {i + 1}: {pred['digit']} (置信度: {pred['confidence']:.2%})\n")

        return student_id, predictions


def main():
    """主函数：识别学号"""
    print("=" * 60)
    print("学号识别系统")
    print("=" * 60)

    # 创建识别器
    recognizer = StudentIDRecognizer()

    image_path = "student_id.jpg"

    try:
        # 识别学号
        student_id, predictions = recognizer.recognize_student_id(image_path)

        print(f"\n最终识别结果: {student_id}")
        print(f"结果已保存到 'recognition_result.txt'")
        print(f"可视化结果已保存到 'student_id_recognition_result.png'")

    except Exception as e:
        print(f"识别过程中出错: {e}")
        print("\n请确保:")
        print("1. 学号图片路径正确")
        print("2. 图片中的数字清晰可辨")
        print("3. 数字之间有一定间隔")
        print("4. 已正确训练模型（运行 train_model.py）")


if __name__ == "__main__":
    main()