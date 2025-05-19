

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os
import cv2
from PIL import Image
import torchvision.transforms.functional as F


# Set the environment variable to avoid OpenMP warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=None

MODEL_PATH = "./maskrcnn_wheel_best.pth"
SCORE_THRESHOLD = 0.5

def get_model_instance_segmentation(num_classes):
    """Create and configure the object detection model"""
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


# Load model in background
def load_model():
    global model
    try:
        model = get_model_instance_segmentation(num_classes=2)
        model.to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Failed to load model: {e}")
    
    return model


def process_frame(frame):
    """Process a frame through the detection model to detect wheels"""
    global model
    load_model()
    
    if model is None:
        return {"boxes": None, "masks": None, "scores": None, "labels": None}
    
    try:
        img_height, img_width = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tensor = F.to_tensor(image_pil)
        
        model.eval()
        with torch.no_grad():
            prediction = model([image_tensor.to(DEVICE)])[0]
        
        boxes = prediction["boxes"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        masks = prediction["masks"].cpu().numpy()
        # 'labels' are the class identifiers where 1 is wheel (background is 0)
        labels = prediction["labels"].cpu().numpy()
        
        # Filter by both score threshold and wheel class (label=1) if available
        # In a binary classification model (background vs wheel), the label should be 1 for wheels
        keep = (scores > SCORE_THRESHOLD) & (labels == 1) if len(labels) > 0 else (scores > SCORE_THRESHOLD)
        
        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep] if len(labels) > 0 else None
        
        # For debugging
        print(f"Detected {len(boxes)} wheels with scores: {scores}")
        
        # If multiple wheels detected, keep only the one with highest confidence
        if len(boxes) > 1:
            best_idx = np.argmax(scores)
            boxes = boxes[best_idx:best_idx+1]  # Keep only the best one
            masks = masks[best_idx:best_idx+1]
            scores = scores[best_idx:best_idx+1]
            if labels is not None:
                labels = labels[best_idx:best_idx+1]
            print(f"Selected best wheel with score {scores[0]}")
        
        return {"boxes": boxes, "masks": masks, "scores": scores, "labels": labels}
    
    except Exception as e:
        print(f"Failed to process frame: {e}")
        return {"boxes": None, "masks": None, "scores": None, "labels": None}


# Start the main loop
if __name__ == "__main__":
   load_model()
   print(process_frame())