import sys
import os
import torch
import boto3
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from segment_anything import sam_model_registry, SamPredictor
from typing import List

# 선행 코드: cd C:\Users\user\Desktop\WOT\grounded-sam
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
from groundingdino.util.inference import Model

# Paths & Device
HOME = os.getcwd()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
S3_BUCKET_NAME = "hanihanibucket"

# Model Paths
GROUNDING_DINO_CONFIG = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
SAM_CHECKPOINT = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")

# Load Models
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT)
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(DEVICE)
sam_predictor = SamPredictor(sam)

# Initialize FastAPI app
app = FastAPI()

# AWS S3 Client
s3 = boto3.client('s3')

# Categories
CATEGORIES = {
    'top': ['blouse', 'shirts', 'jacket', 'coat'],
    'bottom': ['pants', 'skirt', 'dress', 'jean'],
    'shoes': ['shoes']
}

# Enhance class names for detection
def enhance_class_name(classes: List[str]) -> List[str]:
    return [f"all {cls}s" for cls in classes]

# Segment items using SAM
def segment_image(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    masks = []
    for box in boxes:
        mask, score, _ = sam_predictor.predict(box=box, multimask_output=True)
        masks.append(mask[np.argmax(score)])
    return np.array(masks)

# Helper function to combine masks by category
def combine_masks_by_category(
    category: str, titles: List[str], detections: np.ndarray, masks: np.ndarray, image: np.ndarray
) -> np.ndarray:
    white_background = np.ones_like(image) * 255
    indices = np.isin(titles, CATEGORIES[category])

    if not np.any(indices):
        return None  # If no detections for this category, skip

    combined_mask = np.logical_or.reduce(masks[indices], axis=0)
    combined_image = np.where(combined_mask[..., None], image, white_background)

    # Find the bounding box for cropping
    relevant_detections = detections[indices]
    min_x_start = int(np.min(relevant_detections[:, 0]))
    min_y_start = int(np.min(relevant_detections[:, 1]))
    max_x_end = int(np.max(relevant_detections[:, 2]))
    max_y_end = int(np.max(relevant_detections[:, 3]))

    # Crop the combined image
    cropped_image = combined_image[min_y_start:max_y_end, min_x_start:max_x_end, :]
    return cropped_image

# Save segmented image and upload to S3
def save_and_upload_image(category: str, original_filename: str, image: np.ndarray) -> str:
    # Extract original file name without extension
    base_name, ext = os.path.splitext(original_filename)
    file_name = f"{base_name}_{category}{ext}"
    local_path = os.path.join(HOME, file_name)

    # Save the image locally
    cv2.imwrite(local_path, image)

    # Upload to S3 under segmented_img folder
    s3_key = f"segmented_img/{file_name}"
    s3.upload_file(local_path, S3_BUCKET_NAME, s3_key)

    # Generate the S3 URL
    s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
    os.remove(local_path)  # Clean up local storage
    return s3_url

from fastapi import Depends
# API Endpoint to process image
@app.post("/segment")
async def segment_and_upload(
    image: UploadFile = File(...), 
    classes: str = Form(...),
    box_threshold: float = Form(0.35), 
    text_threshold: float = Form(0.25)
):
    # (debugging log) image recognition
    print(f"Received file: {image.filename}")
    print(f"Received classes: {classes}")

    # Load and preprocess image
    try:
        image_data = await image.read()
        print(f"Image data size: {len(image_data)} bytes")
    except Exception as e:
        print(f"Error reading image: {str(e)}")
        return {"error": "Invalid image"}

    np_image = np.frombuffer(image_data, np.uint8)
    image_np = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if image_np is None:
        return {"error": "Failed to decode image"}

    # Run GroundingDINO detection
    detections = grounding_dino_model.predict_with_classes(
        image=image_np, 
        classes=enhance_class_name(classes.split(',')), 
        box_threshold=box_threshold, 
        text_threshold=text_threshold
    )

    # Segment objects with SAM
    masks = segment_image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), detections.xyxy)
    titles = [classes.split(',')[class_id] for class_id in detections.class_id]

    # Upload combined images by category to S3 and return URLs
    s3_urls = []
    for category in CATEGORIES:
        combined_image = combine_masks_by_category(category, titles, detections.xyxy, masks, image_np)
        if combined_image is not None:
            s3_url = save_and_upload_image(category, image.filename, combined_image)
            s3_urls.append(s3_url)

    return {"message": "Image segmented and processed", "urls": s3_urls}
