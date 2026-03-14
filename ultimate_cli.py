"""
================================================================================
ULTIMATE OBJECT DETECTION - CLI VERSION
================================================================================
Professional object detection system for Python 3.13 (No GUI)
Features:
- Webcam detection
- Image detection
- Video processing
- Object counting
- Screenshot capture
- Performance monitoring
================================================================================
"""

import cv2
import numpy as np
import time
import os
import json
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import sys

# ================================================================================
# CONFIGURATION
# ================================================================================

CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Color mapping
COLORS = {
    'high': (0, 255, 0),      # Green
    'medium': (0, 255, 255),  # Yellow
    'low': (0, 165, 255),     # Orange
    'very_low': (0, 0, 255)   # Red
}

# Available models
MODELS = {
    '1': ('Nano (Fastest)', 'yolov8n.pt'),
    '2': ('Small (Balanced)', 'yolov8s.pt'),
    '3': ('Medium (Accurate)', 'yolov8m.pt'),
    '4': ('Large (Most Accurate)', 'yolov8l.pt')
}


# ================================================================================
# OBJECT DETECTION ENGINE
# ================================================================================

class ObjectDetector:
    def __init__(self):
        self.model = None
        self.model_name = 'yolov8n.pt'
        self.confidence = 0.5
        self.stats = {
            'total_detections': 0,
            'objects_count': {},
            'fps_history': []
        }
        self.screenshot_count = 0
        self.load_model()
        
    def load_model(self, model_name=None):
        """Load YOLO model"""
        if model_name:
            self.model_name = model_name
            
        try:
            print(f"\n📥 Loading model: {self.model_name}...")
            self.model = YOLO(self.model_name)
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def detect(self, frame):
        """Detect objects in frame"""
        if self.model is None:
            return frame, []
        
        start_time = time.time()
        
        # Run detection
        results = self.model(frame)
        
        # Process results
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                
                if conf >= self.confidence:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    class_id = int(box.cls[0])
                    class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f'Unknown-{class_id}'
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': class_name
                    })
                    
                    # Update stats
                    self.stats['total_detections'] += 1
                    if class_name in self.stats['objects_count']:
                        self.stats['objects_count'][class_name] += 1
                    else:
                        self.stats['objects_count'][class_name] = 1
        
        # Calculate FPS
        process_time = time.time() - start_time
        fps = 1.0 / process_time if process_time > 0 else 0
        self.stats['fps_history'].append(fps)
        
        return detections, process_time, fps
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class']
            
            # Choose color
            if conf > 0.8:
                color = COLORS['high']
            elif conf > 0.5:
                color = COLORS['medium']
            elif conf > 0.3:
                color = COLORS['low']
            else:
                color = COLORS['very_low']
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f'{class_name}: {conf:.2f}'
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def save_screenshot(self, frame, detections):
        """Save screenshot"""
        self.screenshot_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory
        Path('detections').mkdir(exist_ok=True)
        
        # Save image
        filename = f"detections/detection_{timestamp}_{self.screenshot_count}.jpg"
        cv2.imwrite(filename, frame)
        
        # Save info
        info_file = f"detections/detection_{timestamp}_{self.screenshot_count}.json"
        with open(info_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'confidence_threshold': self.confidence,
                'detections': detections,
                'total_detections': len(detections)
            }, f, indent=2)
        
        return filename
    
    def print_stats(self):
        """Print statistics"""
        print("\n" + "="*60)
        print("📊 DETECTION STATISTICS")
        print("="*60)
        print(f"Total Detections: {self.stats['total_detections']}")
        
        if self.stats['fps_history']:
            avg_fps = sum(self.stats['fps_history'][-100:]) / min(len(self.stats['fps_history']), 100)
            print(f"Average FPS: {avg_fps:.1f}")
        
        print("\nObjects Detected:")
        sorted_objects = sorted(self.stats['objects_count'].items(), key=lambda x: x[1], reverse=True)
        for obj, count in sorted_objects[:10]:
            print(f"  • {obj}: {count}")
        print("="*60)
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_detections': 0,
            'objects_count': {},
            'fps_history': []
        }
        print("✅ Statistics reset")


# ================================================================================
# MAIN APPLICATION
# ================================================================================

class ObjectDetectionApp:
    def __init__(self):
        self.detector = ObjectDetector()
        
    def show_banner(self):
        """Show welcome banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════════╗
║              ULTIMATE OBJECT DETECTION SYSTEM v2.0               ║
║                         CLI VERSION                              ║
║                   Compatible with Python 3.13                    ║
╚═══════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def show_menu(self):
        """Show main menu"""
        menu = """
╔═══════════════════════════════════════════════════════════════════╗
║                           MAIN MENU                               ║
╠═══════════════════════════════════════════════════════════════════╣
║  1. 🎥 Webcam Detection (Real-time)                              ║
║  2. 🖼️  Image File Detection                                      ║
║  3. 🎬 Video File Detection                                       ║
║  4. ⚙️  Change Model                                              ║
║  5. 📊 View Statistics                                            ║
║  6. 🔄 Reset Statistics                                           ║
║  7. ❌ Exit                                                       ║
╚═══════════════════════════════════════════════════════════════════╝
        """
        print(menu)
    
    def webcam_mode(self):
        """Webcam detection mode"""
        print("\n🎥 Starting webcam detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  '+' - Increase confidence")
        print("  '-' - Decrease confidence")
        print("-"*60)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect objects
            detections, proc_time, fps = self.detector.detect(frame)
            
            # Draw detections
            frame = self.detector.draw_detections(frame, detections)
            
            # Add info text
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {self.detector.confidence:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Objects: {len(detections)}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Object Detection - Press "q" to quit', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = self.detector.save_screenshot(frame, detections)
                print(f"✅ Screenshot saved: {filename}")
            elif key == ord('+') or key == ord('='):
                self.detector.confidence = min(1.0, self.detector.confidence + 0.05)
                print(f"📈 Confidence: {self.detector.confidence:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.detector.confidence = max(0.1, self.detector.confidence - 0.05)
                print(f"📉 Confidence: {self.detector.confidence:.2f}")
            
            # Show stats every 100 frames
            if frame_count % 100 == 0:
                print(f"📊 Processed {frame_count} frames | FPS: {fps:.1f}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Webcam detection stopped")
        self.detector.print_stats()
    
    def image_mode(self):
        """Image detection mode"""
        filename = input("\n📁 Enter image filename: ").strip()
        
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            return
        
        # Load image
        frame = cv2.imread(filename)
        
        # Detect objects
        detections, proc_time, fps = self.detector.detect(frame)
        
        # Draw detections
        result = self.detector.draw_detections(frame.copy(), detections)
        
        # Show results
        print(f"\n✅ Found {len(detections)} objects")
        print(f"⏱️  Processing time: {proc_time*1000:.1f}ms")
        
        # List detections
        if detections:
            print("\nDetected objects:")
            for i, det in enumerate(detections, 1):
                print(f"  {i}. {det['class']}: {det['confidence']:.2f}")
        
        # Show image
        cv2.imshow('Original', frame)
        cv2.imshow('Detection Result', result)
        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save option
        save = input("\nSave result? (y/n): ").strip().lower()
        if save == 'y':
            output = f"detected_{filename}"
            cv2.imwrite(output, result)
            print(f"✅ Saved as: {output}")
    
    def video_mode(self):
        """Video detection mode"""
        filename = input("\n📁 Enter video filename: ").strip()
        
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            return
        
        cap = cv2.VideoCapture(filename)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo Info:")
        print(f"  FPS: {fps}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Total frames: {total_frames}")
        
        # Ask to save
        save_video = input("\nSave output video? (y/n): ").strip().lower()
        
        if save_video == 'y':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output_detection.mp4', fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print("\nProcessing video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect objects
            detections, proc_time, current_fps = self.detector.detect(frame)
            
            # Draw detections
            frame = self.detector.draw_detections(frame, detections)
            
            if save_video == 'y':
                out.write(frame)
            
            # Show progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / frame_count) * (total_frames - frame_count)
                print(f"  Progress: {progress:.1f}% | FPS: {current_fps:.1f} | ETA: {eta:.1f}s", end='\r')
            
            # Show frame (optional, slows down processing)
            # cv2.imshow('Processing', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        cap.release()
        if save_video == 'y':
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Processing complete! Processed {frame_count} frames")
        print(f"⏱️  Total time: {time.time() - start_time:.1f}s")
        self.detector.print_stats()
    
    def change_model(self):
        """Change YOLO model"""
        print("\nAvailable Models:")
        for key, (name, _) in MODELS.items():
            print(f"  {key}. {name}")
        
        choice = input("\nSelect model (1-4): ").strip()
        
        if choice in MODELS:
            name, model_file = MODELS[choice]
            if self.detector.load_model(model_file):
                print(f"✅ Model changed to: {name}")
        else:
            print("❌ Invalid choice")
    
    def run(self):
        """Run the application"""
        self.show_banner()
        
        while True:
            self.show_menu()
            choice = input("\n👉 Enter your choice: ").strip()
            
            if choice == '1':
                self.webcam_mode()
            elif choice == '2':
                self.image_mode()
            elif choice == '3':
                self.video_mode()
            elif choice == '4':
                self.change_model()
            elif choice == '5':
                self.detector.print_stats()
            elif choice == '6':
                self.detector.reset_stats()
            elif choice == '7':
                print("\n👋 Thank you for using Object Detection System!")
                self.detector.print_stats()
                break
            else:
                print("❌ Invalid choice!")


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    app = ObjectDetectionApp()
    app.run()