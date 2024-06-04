import os
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

# 전역 변수 설정
DATA_DIR = Path("")  # 데이터 디렉토리 경로 설정
IMAGE_SIZE = 512

# JSON 파일 로드 및 유효한 라벨 ID 가져오기
with open(DATA_DIR / "label_descriptions.json") as f:
    label_descriptions = json.load(f)

# 유효한 라벨 ID와 이름 가져오기
valid_labels = {x['id']: x['name'] for x in label_descriptions['categories']}
label_names = list(valid_labels.values())
NUM_CLASSES = len(valid_labels)  # 유효한 클래스 수
print(NUM_CLASSES)
# CSV 파일 로드 및 전처리
segment_df = pd.read_csv(DATA_DIR / "train.csv")

# 유효한 라벨 ID로 필터링
segment_df['CategoryId'] = segment_df['ClassId'].str.split('_').str[0].astype(int)
valid_category_ids = set(valid_labels.keys())
segment_df = segment_df[segment_df['CategoryId'].isin(valid_category_ids)]

print(f"Total segments after filtering: {len(segment_df)}")

if len(segment_df) == 0:
    raise ValueError("No valid segments found after filtering. Please check the label_descriptions.json file and ensure that it contains valid category IDs.")

image_df = segment_df.groupby('ImageId')[['EncodedPixels', 'CategoryId']].agg(lambda x: list(x))
size_df = segment_df.groupby('ImageId')[['Height', 'Width']].mean()
image_df = image_df.join(size_df, on='ImageId')

print(f"Total images after filtering: {len(image_df)}")

if len(image_df) == 0:
    raise ValueError("No valid images found after filtering. Please check the label_descriptions.json file and ensure that it contains valid category IDs.")

# 이미지 리사이즈 함수 정의
def resize_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} does not exist")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error loading image {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img


# Custom Dataset 클래스 정의
class FashionDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.image_ids = df.index.tolist()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = DATA_DIR / 'train' / f'{image_id}'
        try:
            img = resize_image(str(img_path))
        except (FileNotFoundError, ValueError) as e:
            print(e)
            return self.__getitem__((idx + 1) % len(self))  # Load next image if error

        info = self.df.loc[image_id]
        annotations = info['EncodedPixels']
        labels = info['CategoryId']
        height = int(info['Height'])
        width = int(info['Width'])

        mask, labels = self.load_mask(annotations, labels, height, width)

        img = T.ToTensor()(img)  # 이미지를 텐서로 변환

        target = {}
        target['boxes'] = self.get_boxes(mask)
        target['labels'] = torch.tensor(labels, dtype=torch.int64)
        target['masks'] = mask
        target['image_id'] = torch.tensor([idx])

        # 유효한 바운딩 박스 필터링
        valid_indices = []
        for i, box in enumerate(target['boxes']):
            if (box[2] > box[0]) and (box[3] > box[1]):
                valid_indices.append(i)

        target['boxes'] = target['boxes'][valid_indices]
        target['labels'] = target['labels'][valid_indices]
        target['masks'] = target['masks'][:, :, valid_indices]

        return img, target

    def load_mask(self, annotations, labels, height, width):
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(annotations)), dtype=np.uint8)
        labels_out = []

        for m, (annotation, label) in enumerate(zip(annotations, labels)):
            sub_mask = np.full(height * width, 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]

            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel + annotation[2 * i + 1]] = 1

            sub_mask = sub_mask.reshape((height, width), order='F')
            sub_mask = cv2.resize(sub_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

            mask[:, :, m] = sub_mask
            labels_out.append(int(label))

        return torch.as_tensor(mask, dtype=torch.uint8), labels_out

    def get_boxes(self, masks):
        num_objs = masks.shape[-1]
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[:, :, i])
            if pos[0].size == 0 or pos[1].size == 0:  # 빈 마스크 무시
                continue
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        return torch.as_tensor(boxes, dtype=torch.float32)


def collate_fn(batch):
    return tuple(zip(*batch))


# Mask R-CNN 모델 설정
def get_model_instance_segmentation(num_classes):
    # Mask R-CNN 모델을 불러옴 (ResNet-50 백본 사용)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # 분류기 예측기 수를 새로운 값으로 변경 (배경 클래스 포함)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # 마스크 예측기를 새로운 값으로 변경
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


# 시각화 함수 정의
def visualize_sample(img, target):
    img = img.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    current_axis = plt.gca()

    masks = target['masks'].numpy()
    boxes = target['boxes'].numpy()
    labels = target['labels'].numpy()

    for i in range(masks.shape[-1]):
        mask = masks[:, :, i]
        color = np.random.rand(3)

        # Create a mask with random color
        mask_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for j in range(3):
            mask_image[:, :, j] = mask * int(color[j] * 255)

        # Overlay the mask on the original image
        overlay_image = Image.fromarray(mask_image).convert("RGBA")
        image_pil = Image.fromarray((img * 255).astype(np.uint8)).convert("RGBA")

        # Ensure the images have the same size
        overlay_image = overlay_image.resize(image_pil.size, resample=Image.BILINEAR)

        blended_image = Image.blend(image_pil, overlay_image, alpha=0.5)

        plt.imshow(blended_image)

        # Draw bounding box
        box = boxes[i]
        xmin, ymin, xmax, ymax = box
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                             fill=False, edgecolor=color, linewidth=2))
        current_axis.text(xmin, ymin, str(labels[i]), bbox={'facecolor': color, 'alpha': 0.5})

    plt.axis('off')
    plt.show()


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_labels = output['labels'].cpu().numpy()
                true_labels = target['labels'].cpu().numpy()

                # 각 객체별로 라벨을 비교
                for pred_label, true_label in zip(pred_labels, true_labels):
                    if pred_label == true_label:
                        correct += 1
                    total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


if __name__ == '__main__':
    # 데이터셋 생성
    dataset = FashionDataset(image_df)

    # 데이터셋을 train과 test로 나누기
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # DataLoader 정의
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # 모델 학습 설정
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(NUM_CLASSES + 1)  # 배경 클래스 포함
    model.to(device)

    # 옵티마이저 설정
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)

    # 학습 루프
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        i = 0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            i += 1
            print(f"Epoch {epoch + 1}, Iteration {i}, Loss: {losses.item()}")
            if i%5000==0:
                torch.save(model.state_dict(), str(epoch)+str(i)+"maskrcnn_model.pth")
        # 에포크가 끝날 때마다 모델을 평가하고 정확도를 출력
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}, Accuracy: {accuracy:.4f}")

    print("Training completed")

    # 모델 저장

