import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime


class BicycleDetector:
    """校园共享单车检测器"""

    def __init__(self, model_path=None, confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold

        print("初始化自行车检测器...")
        print(f"使用设备: {self.device}")
        print(f"置信度阈值: {confidence_threshold}")

        # 加载模型
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # 预处理变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        print("检测器初始化完成！")

    def load_model(self, model_path=None):
        """加载训练好的模型"""
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        if model_path and os.path.exists(model_path):
            print(f"加载自定义训练模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
        else:
            print("使用 COCO 预训练模型 (自行车类别已包含)")
        return model

    def detect(self, image_path):
        """检测图像中的自行车"""
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            return [], None

        # 读取图像
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)

        # 预处理
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            predictions = self.model(image_tensor)[0]

        # 处理检测结果
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            # COCO bicycle label = 2
            if label == 2 and score >= self.confidence_threshold:
                x1, y1, x2, y2 = box.astype(int)
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(score),
                    'class': 'bicycle',
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'center': ((x1 + x2)//2, (y1 + y2)//2),
                    'area': (x2 - x1) * (y2 - y1)
                })

        return detections, original_image

    def draw_on_image(self, image, detections):
        """在图像上绘制检测框"""
        result_image = image.copy()
        for i, det in enumerate(detections):
            bbox = det['bbox']
            cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            label = f"Bicycle {i+1}: {det['confidence']:.2f}"
            cv2.putText(result_image, label, (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        return result_image

    def visualize(self, image, detections, save_path=None):
        """Matplotlib 可视化"""
        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(image)
        for det in detections:
            bbox = det['bbox']
            rect = patches.Rectangle((bbox[0], bbox[1]),
                                     bbox[2]-bbox[0],
                                     bbox[3]-bbox[1],
                                     linewidth=2,
                                     edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1]-5, f"{det['class']} {det['confidence']:.2f}",
                    color='yellow', fontsize=10, fontweight='bold')

        plt.axis('off')
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化结果保存到: {save_path}")
        plt.show()

    def save_results(self, detections, image_path, output_dir='detection_results'):
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_result.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"检测时间: {datetime.now()}\n")
            f.write(f"检测到 {len(detections)} 辆自行车\n\n")
            for i, det in enumerate(detections):
                bbox = det['bbox']
                f.write(f"Bicycle {i+1}: bbox={bbox}, confidence={det['confidence']:.4f}\n")
        print(f"检测结果已保存到: {txt_path}")
        return txt_path


def main():
    img_path = 'test_bicycle.jpg'

    detector = BicycleDetector(confidence_threshold=0.5)
    detections, image = detector.detect(img_path)

    if image is None:
        return

    print(f"检测到 {len(detections)} 辆自行车")
    for i, det in enumerate(detections, 1):
        print(f"Bicycle {i}: bbox={det['bbox']}, confidence={det['confidence']:.2f}")

    # 绘制检测框
    result_image = detector.draw_on_image(image, detections)
    cv2.imwrite('detected_result.png', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    print("检测结果图片已保存: detected_result.png")

    # 可视化
    detector.visualize(image, detections, save_path='visualization_result.png')

    # 保存结果到文本
    detector.save_results(detections, img_path)


if __name__ == "__main__":
    main()
