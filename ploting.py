import torch
import json
import torchvision.transforms as T
import torchvision
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
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


DATA_DIR = Path("")  # 데이터 디렉토리 경로 설정
IMAGE_SIZE = 512
with open(DATA_DIR / "label_descriptions.json") as f:
    label_descriptions = json.load(f)

# 유효한 라벨 ID와 이름 가져오기
valid_labels = {x['id']: x['name'] for x in label_descriptions['categories']}
label_names = list(valid_labels.values())
NUM_CLASSES = len(valid_labels)  # 유효한 클래스 수

# 모델 불러오기 함수
def load_model(model_path, num_classes):
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# 이미지 리사이즈 함수
def resize_image(image_path, size):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error loading image {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


# 마스크를 세밀하게 조정하는 함수
def refine_masks(masks):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    return masks


# 결과 시각화 함수
def visualize_instance_masks(image, masks, labels, scores, score_threshold=0.5):
    for i in range(masks.shape[0]):
        if scores[i] < score_threshold:
            continue
        mask = masks[i, :, :]
        color = np.random.randint(0, 255, (1, 3)).tolist()[0]
        mask_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for c in range(3):
            mask_image[:, :, c] = mask * color[c]*10000

        plt.figure(figsize=(10, 10))
        plt.imshow(mask_image)
        plt.title(f'Label: {labels[i]}, Score: {scores[i]:.2f}')
        plt.axis('off')
        plt.show()


# 모델 경로와 이미지 경로 설정
model_path = "05000maskrcnn_model.pth"
image_paths = ["testt.jpg"]  # 테스트할 이미지 경로 리스트

# 모델 로드
num_classes = len(valid_labels) + 1  # 배경 클래스 포함
model = load_model(model_path, num_classes)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
def segment_and_plot_image(outputs, score_threshold=0.5):

    # 결과 시각화
    output = outputs[0]
    masks = (output['masks'] > score_threshold).squeeze().cpu().numpy()
    labels = output['labels'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    boxes = output['boxes'].cpu().numpy()

    # 유효한 마스크 필터링
    valid_masks = [masks[i] for i in range(masks.shape[0]) if scores[i] > score_threshold]
    valid_labels = [labels[i] for i in range(labels.shape[0]) if scores[i] > score_threshold]
    valid_boxes = [boxes[i] for i in range(boxes.shape[0]) if scores[i] > score_threshold]

    # 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    current_axis = plt.gca()

    for i, mask in enumerate(valid_masks):
        color = np.random.rand(3)
        mask_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for j in range(3):
            mask_image[:, :, j] = mask * int(color[j] * 255)

        overlay_image = Image.fromarray(mask_image).convert("RGBA")
        image_pil = Image.fromarray(img).convert("RGBA")
        blended_image = Image.blend(image_pil, overlay_image, alpha=0.5)

        plt.imshow(blended_image)

        xmin, ymin, xmax, ymax = valid_boxes[i]
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                             fill=False, edgecolor=color, linewidth=2))
        current_axis.text(xmin, ymin, str(valid_labels[i]), bbox={'facecolor': color, 'alpha': 0.5})

    plt.axis('off')
    plt.show()
# 테스트 이미지에 대해 segmentation 수행 및 시각화
for image_path in image_paths:
    img = resize_image(image_path, IMAGE_SIZE)
    img_tensor = T.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
    segment_and_plot_image(outputs,0.5)
    output = outputs[0]
    masks = output['masks'].squeeze().cpu().numpy()
    print(masks.shape)
    labels = output['labels'].cpu().numpy()
    print(labels.shape)
    scores = output['scores'].cpu().numpy()
    print(scores.shape)
    # 유효한 마스크 필터링 및 시각화
    #masks = refine_masks(masks)
    visualize_instance_masks(img, masks, labels, scores, score_threshold=0.5)