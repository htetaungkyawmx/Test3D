"""
Utility functions for face recognition system
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
from typing import List, Tuple, Optional
import hashlib

def preprocess_face_image(image_path: str, target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    Preprocess face image for better recognition
    
    Args:
        image_path: Path to face image
        target_size: Target size for output image
    
    Returns:
        Preprocessed image
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    return img

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU score
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def get_face_quality_score(face_img: np.ndarray) -> float:
    """
    Calculate quality score for a face image
    
    Args:
        face_img: Face image
    
    Returns:
        Quality score (0-1)
    """
    # Convert to grayscale
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
    
    # Calculate Laplacian variance (blur detection)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Normalize to 0-1 range
    blur_score = min(1.0, laplacian_var / 100.0)
    
    # Calculate brightness
    brightness = np.mean(gray) / 255.0
    brightness_score = 1.0 - abs(brightness - 0.5) * 2
    
    # Combine scores
    quality = (blur_score + brightness_score) / 2
    
    return quality

def save_unknown_face(face_img: np.ndarray, timestamp: str = None) -> str:
    """
    Save unknown face for later review
    
    Args:
        face_img: Face image
        timestamp: Optional timestamp
    
    Returns:
        Path to saved image
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory if not exists
    unknown_dir = "captured_faces/unknown"
    if not os.path.exists(unknown_dir):
        os.makedirs(unknown_dir)
    
    # Generate filename
    hash_val = hashlib.md5(face_img.tobytes()).hexdigest()[:8]
    filename = f"unknown_{timestamp}_{hash_val}.jpg"
    filepath = os.path.join(unknown_dir, filename)
    
    # Save image
    cv2.imwrite(filepath, face_img)
    
    return filepath

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_config(config_path: str, config: dict):
    """Save configuration to JSON file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def calculate_recognition_stats(recognitions: List[dict]) -> dict:
    """
    Calculate statistics from recognition results
    
    Args:
        recognitions: List of recognition results
    
    Returns:
        Statistics dictionary
    """
    if not recognitions:
        return {}
    
    stats = {
        'total': len(recognitions),
        'known': sum(1 for r in recognitions if r['name'] != 'Unknown'),
        'unknown': sum(1 for r in recognitions if r['name'] == 'Unknown'),
        'avg_confidence': 0,
        'top_faces': {}
    }
    
    # Calculate average confidence for known faces
    known_confidences = [r['confidence'] for r in recognitions if r['name'] != 'Unknown']
    if known_confidences:
        stats['avg_confidence'] = sum(known_confidences) / len(known_confidences)
    
    # Count frequency of each face
    face_counts = {}
    for r in recognitions:
        name = r['name']
        if name != 'Unknown':
            face_counts[name] = face_counts.get(name, 0) + 1
    
    # Get top 5 faces
    stats['top_faces'] = dict(sorted(face_counts.items(), key=lambda x: x[1], reverse=True)[:5])
    
    return stats

def create_video_writer(output_path: str, fps: int, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    """
    Create video writer for saving output
    
    Args:
        output_path: Path for output video
        fps: Frames per second
        frame_size: (width, height)
    
    Returns:
        VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)

def overlay_text(frame: np.ndarray, text: str, position: Tuple[int, int], 
                 font_scale: float = 0.6, color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2, background: bool = True) -> np.ndarray:
    """
    Overlay text with optional background
    
    Args:
        frame: Input frame
        text: Text to overlay
        position: (x, y) position
        font_scale: Font size
        color: Text color
        thickness: Text thickness
        background: Whether to add background rectangle
    
    Returns:
        Frame with text overlay
    """
    if background:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        cv2.rectangle(frame, (position[0] - 5, position[1] - text_size[1] - 5),
                     (position[0] + text_size[0] + 5, position[1] + 5), color, -1)
        text_color = (0, 0, 0)
    else:
        text_color = color
    
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, text_color, thickness)
    
    return frame