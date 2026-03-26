"""
================================================================================
ULTIMATE OBJECT DETECTION APPLICATION
================================================================================
A professional-grade object detection system with:
- Real-time webcam detection
- Image file detection
- Video file processing
- Object counting
- Screenshot capture
- Performance monitoring
- Beautiful UI with stats display
- Multiple model support
- Customizable settings
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
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import queue
import sys

# ================================================================================
# CONFIGURATION
# ================================================================================

class Config:
    """Configuration settings for the application"""
    
    # COCO dataset class names (80 classes that YOLO can detect)
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
    
    # Color mapping for different confidence levels
    CONFIDENCE_COLORS = {
        'high': (0, 255, 0),      # Green - 80-100%
        'medium': (0, 255, 255),  # Yellow - 50-80%
        'low': (0, 165, 255),     # Orange - 30-50%
        'very_low': (0, 0, 255)   # Red - below 30%
    }
    
    # Available YOLO models
    MODELS = {
        '1: Nano (Fastest)': 'yolov8n.pt',
        '2: Small (Balanced)': 'yolov8s.pt',
        '3: Medium (Accurate)': 'yolov8m.pt',
        '4: Large (Most Accurate)': 'yolov8l.pt'
    }
    
    # Default settings
    DEFAULT_CONFIDENCE = 0.5
    DEFAULT_MODEL = 'yolov8n.pt'
    SAVE_DIR = 'detections'
    CONFIG_FILE = 'object_detection_config.json'


# ================================================================================
# OBJECT DETECTION ENGINE
# ================================================================================

class ObjectDetectionEngine:
    """Core detection engine that handles all object detection operations"""
    
    def __init__(self):
        self.model = None
        self.model_name = Config.DEFAULT_MODEL
        self.confidence_threshold = Config.DEFAULT_CONFIDENCE
        self.is_running = False
        self.stats = {
            'total_detections': 0,
            'fps': 0,
            'processing_time': 0,
            'objects_count': {},
            'start_time': None
        }
        self.screenshot_counter = 0
        self.load_model()
        
    def load_model(self, model_name=None):
        """Load YOLO model"""
        if model_name:
            self.model_name = model_name
            
        try:
            print(f"Loading model: {self.model_name}")
            self.model = YOLO(self.model_name)
            print("✓ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def detect_objects(self, frame):
        """Perform object detection on a frame"""
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
                confidence = float(box.conf[0])
                
                if confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    class_id = int(box.cls[0])
                    class_name = Config.CLASS_NAMES[class_id] if class_id < len(Config.CLASS_NAMES) else f'Unknown-{class_id}'
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    })
                    
                    # Update stats
                    self.stats['total_detections'] += 1
                    if class_name in self.stats['objects_count']:
                        self.stats['objects_count'][class_name] += 1
                    else:
                        self.stats['objects_count'][class_name] = 1
        
        # Update performance stats
        processing_time = time.time() - start_time
        self.stats['processing_time'] = processing_time
        self.stats['fps'] = 1.0 / processing_time if processing_time > 0 else 0
        
        return frame, detections
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Choose color based on confidence
            if confidence > 0.8:
                color = Config.CONFIDENCE_COLORS['high']
            elif confidence > 0.5:
                color = Config.CONFIDENCE_COLORS['medium']
            elif confidence > 0.3:
                color = Config.CONFIDENCE_COLORS['low']
            else:
                color = Config.CONFIDENCE_COLORS['very_low']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f'{class_name}: {confidence:.2f}'
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw background for text
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def draw_stats(self, frame):
        """Draw statistics on frame"""
        y_offset = 30
        
        # Title
        cv2.putText(frame, "OBJECT DETECTION STATS", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Model info
        cv2.putText(frame, f"Model: {self.model_name}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
        
        # Confidence threshold
        cv2.putText(frame, f"Confidence: {self.confidence_threshold:.2f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
        
        # FPS
        cv2.putText(frame, f"FPS: {self.stats['fps']:.1f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 20
        
        # Processing time
        cv2.putText(frame, f"Time: {self.stats['processing_time']*1000:.1f}ms", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
        
        # Total detections
        cv2.putText(frame, f"Total: {self.stats['total_detections']}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 25
        
        # Current objects
        cv2.putText(frame, "Current Objects:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 20
        
        # Show top 5 objects in current frame
        current_objects = {}
        for det in detections:
            if det['class_name'] in current_objects:
                current_objects[det['class_name']] += 1
            else:
                current_objects[det['class_name']] = 1
        
        for obj, count in list(current_objects.items())[:5]:
            cv2.putText(frame, f"  {obj}: {count}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15
        
        # Instructions
        y_offset = frame.shape[0] - 100
        cv2.putText(frame, "COMMANDS:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(frame, "Q: Quit | S: Screenshot | C: Confidence +/-", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += 15
        cv2.putText(frame, "R: Reset Stats | M: Change Model | P: Show Report", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def save_screenshot(self, frame, detections):
        """Save current frame as screenshot"""
        self.screenshot_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create save directory if not exists
        Path(Config.SAVE_DIR).mkdir(exist_ok=True)
        
        # Save image
        filename = f"{Config.SAVE_DIR}/detection_{timestamp}_{self.screenshot_counter}.jpg"
        cv2.imwrite(filename, frame)
        
        # Save detection info
        info_file = f"{Config.SAVE_DIR}/detection_{timestamp}_{self.screenshot_counter}.json"
        with open(info_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'model': self.model_name,
                'confidence_threshold': self.confidence_threshold,
                'detections': detections
            }, f, indent=2)
        
        return filename
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_detections': 0,
            'fps': 0,
            'processing_time': 0,
            'objects_count': {},
            'start_time': datetime.now().isoformat()
        }
        print("✓ Statistics reset")
    
    def generate_report(self):
        """Generate detection report"""
        report = f"""
================================================================================
OBJECT DETECTION REPORT
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {self.model_name}
Confidence Threshold: {self.confidence_threshold}

STATISTICS:
- Total Detections: {self.stats['total_detections']}
- Average FPS: {self.stats['fps']:.1f}
- Average Processing Time: {self.stats['processing_time']*1000:.1f}ms

OBJECTS DETECTED:
"""
        
        # Sort objects by count
        sorted_objects = sorted(self.stats['objects_count'].items(), key=lambda x: x[1], reverse=True)
        for obj, count in sorted_objects:
            report += f"  {obj}: {count}\n"
        
        report += "=" * 80
        return report


# ================================================================================
# GUI APPLICATION
# ================================================================================

class ObjectDetectionApp:
    """Main GUI application"""
    
    def __init__(self):
        self.engine = ObjectDetectionEngine()
        self.running = False
        self.current_mode = 'webcam'  # 'webcam', 'image', 'video'
        self.video_capture = None
        self.video_writer = None
        self.frame_queue = queue.Queue()
        
        self.create_gui()
        
    def create_gui(self):
        """Create main GUI window"""
        self.root = tk.Tk()
        self.root.title("Ultimate Object Detection System")
        self.root.geometry("900x700")
        
        # Style
        self.root.configure(bg='#2b2b2b')
        
        # Title
        title_label = tk.Label(
            self.root, 
            text="🔍 ULTIMATE OBJECT DETECTION SYSTEM", 
            font=("Arial", 20, "bold"),
            fg='white',
            bg='#2b2b2b'
        )
        title_label.pack(pady=10)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_frame, bg='#3c3c3c', width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Control sections
        self.create_model_section(left_panel)
        self.create_mode_section(left_panel)
        self.create_detection_section(left_panel)
        self.create_stats_section(left_panel)
        
        # Right panel - Display
        right_panel = tk.Frame(main_frame, bg='#1e1e1e')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Video display area
        self.video_label = tk.Label(
            right_panel,
            text="No video source selected",
            bg='#1e1e1e',
            fg='white',
            font=("Arial", 14)
        )
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bg='#1e1e1e',
            fg='white',
            anchor='w',
            font=("Arial", 10)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Start GUI update loop
        self.update_gui()
        
    def create_model_section(self, parent):
        """Create model selection section"""
        frame = tk.LabelFrame(parent, text="Model Selection", bg='#3c3c3c', fg='white', font=("Arial", 12, "bold"))
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Model dropdown
        self.model_var = tk.StringVar(value=list(Config.MODELS.keys())[0])
        model_menu = ttk.Combobox(frame, textvariable=self.model_var, values=list(Config.MODELS.keys()))
        model_menu.pack(fill=tk.X, padx=5, pady=5)
        model_menu.bind('<<ComboboxSelected>>', self.change_model)
        
        # Load model button
        tk.Button(frame, text="Load Model", command=self.load_selected_model,
                 bg='#4a4a4a', fg='white').pack(fill=tk.X, padx=5, pady=5)
        
    def create_mode_section(self, parent):
        """Create mode selection section"""
        frame = tk.LabelFrame(parent, text="Input Source", bg='#3c3c3c', fg='white', font=("Arial", 12, "bold"))
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(frame, text="Webcam", command=self.start_webcam,
                 bg='#4a4a4a', fg='white').pack(fill=tk.X, padx=5, pady=2)
        tk.Button(frame, text="Image File", command=self.load_image,
                 bg='#4a4a4a', fg='white').pack(fill=tk.X, padx=5, pady=2)
        tk.Button(frame, text="Video File", command=self.load_video,
                 bg='#4a4a4a', fg='white').pack(fill=tk.X, padx=5, pady=2)
        tk.Button(frame, text="Stop", command=self.stop_detection,
                 bg='#8b0000', fg='white').pack(fill=tk.X, padx=5, pady=5)
        
    def create_detection_section(self, parent):
        """Create detection controls section"""
        frame = tk.LabelFrame(parent, text="Detection Controls", bg='#3c3c3c', fg='white', font=("Arial", 12, "bold"))
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Confidence threshold
        tk.Label(frame, text="Confidence Threshold:", bg='#3c3c3c', fg='white').pack(anchor='w', padx=5, pady=2)
        self.confidence_var = tk.DoubleVar(value=Config.DEFAULT_CONFIDENCE)
        confidence_scale = tk.Scale(frame, from_=0.1, to=1.0, resolution=0.05, 
                                    orient=tk.HORIZONTAL, variable=self.confidence_var,
                                    bg='#3c3c3c', fg='white', highlightbackground='#3c3c3c')
        confidence_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # Control buttons
        button_frame = tk.Frame(frame, bg='#3c3c3c')
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(button_frame, text="Screenshot", command=self.take_screenshot,
                 bg='#4a4a4a', fg='white').pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(button_frame, text="Reset Stats", command=self.reset_stats,
                 bg='#4a4a4a', fg='white').pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        tk.Button(frame, text="Generate Report", command=self.show_report,
                 bg='#4a4a4a', fg='white').pack(fill=tk.X, padx=5, pady=2)
        
    def create_stats_section(self, parent):
        """Create statistics display section"""
        frame = tk.LabelFrame(parent, text="Live Statistics", bg='#3c3c3c', fg='white', font=("Arial", 12, "bold"))
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Stats display
        self.stats_text = tk.Text(frame, bg='#2b2b2b', fg='white', height=10, width=30)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def change_model(self, event=None):
        """Handle model change"""
        selected = self.model_var.get()
        model_file = Config.MODELS[selected]
        self.engine.load_model(model_file)
        self.update_status(f"Model changed to {selected}")
        
    def load_selected_model(self):
        """Load selected model"""
        selected = self.model_var.get()
        model_file = Config.MODELS[selected]
        if self.engine.load_model(model_file):
            self.update_status(f"✓ Model loaded: {selected}")
        else:
            self.update_status(f"❌ Failed to load model: {selected}")
    
    def start_webcam(self):
        """Start webcam detection"""
        self.stop_detection()
        self.current_mode = 'webcam'
        self.video_capture = cv2.VideoCapture(0)
        self.running = True
        self.update_status("Webcam started")
        self.process_video()
    
    def load_image(self):
        """Load and process image file"""
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if filename:
            self.stop_detection()
            self.current_mode = 'image'
            
            # Load and process image
            frame = cv2.imread(filename)
            frame, detections = self.engine.detect_objects(frame)
            frame = self.engine.draw_detections(frame, detections)
            frame = self.engine.draw_stats(frame)
            
            # Display image
            self.display_frame(frame)
            self.update_status(f"Image processed: {os.path.basename(filename)}")
    
    def load_video(self):
        """Load and process video file"""
        filename = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if filename:
            self.stop_detection()
            self.current_mode = 'video'
            self.video_capture = cv2.VideoCapture(filename)
            self.running = True
            self.update_status(f"Video loaded: {os.path.basename(filename)}")
            self.process_video()
    
    def stop_detection(self):
        """Stop current detection"""
        self.running = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.update_status("Detection stopped")
    
    def take_screenshot(self):
        """Take screenshot of current frame"""
        if hasattr(self, 'current_frame'):
            filename = self.engine.save_screenshot(self.current_frame, self.current_detections)
            self.update_status(f"✓ Screenshot saved: {filename}")
    
    def reset_stats(self):
        """Reset statistics"""
        self.engine.reset_stats()
        self.update_status("Statistics reset")
    
    def show_report(self):
        """Show detection report"""
        report = self.engine.generate_report()
        
        # Create report window
        report_window = tk.Toplevel(self.root)
        report_window.title("Detection Report")
        report_window.geometry("600x400")
        
        # Text widget
        text_widget = tk.Text(report_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert('1.0', report)
        
        # Save button
        tk.Button(report_window, text="Save Report", 
                 command=lambda: self.save_report(report)).pack(pady=5)
    
    def save_report(self, report):
        """Save report to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                f.write(report)
            self.update_status(f"Report saved: {filename}")
    
    def process_video(self):
        """Process video/webcam frames"""
        if not self.running or not self.video_capture:
            return
        
        ret, frame = self.video_capture.read()
        if ret:
            # Update confidence threshold
            self.engine.confidence_threshold = self.confidence_var.get()
            
            # Process frame
            frame, detections = self.engine.detect_objects(frame)
            frame = self.engine.draw_detections(frame, detections)
            frame = self.engine.draw_stats(frame)
            
            # Store for screenshot
            self.current_frame = frame
            self.current_detections = detections
            
            # Display
            self.display_frame(frame)
            
            # Update stats display
            self.update_stats_display()
        
        # Continue processing
        if self.running:
            self.root.after(10, self.process_video)
    
    def display_frame(self, frame):
        """Display frame in GUI"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for display
        height, width = frame_rgb.shape[:2]
        max_height = 480
        max_width = 640
        
        if height > max_height or width > max_width:
            scale = min(max_height/height, max_width/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
        
        # Convert to PhotoImage
        from PIL import Image, ImageTk
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image)
        
        # Update label
        self.video_label.config(image=photo)
        self.video_label.image = photo
    
    def update_stats_display(self):
        """Update statistics display"""
        self.stats_text.delete('1.0', tk.END)
        
        stats = self.engine.stats
        text = f"Total Detections: {stats['total_detections']}\n"
        text += f"FPS: {stats['fps']:.1f}\n"
        text += f"Processing: {stats['processing_time']*1000:.1f}ms\n"
        text += "\nTop Objects:\n"
        
        sorted_objects = sorted(stats['objects_count'].items(), key=lambda x: x[1], reverse=True)[:10]
        for obj, count in sorted_objects:
            text += f"  {obj}: {count}\n"
        
        self.stats_text.insert('1.0', text)
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=f"  {message}")
    
    def update_gui(self):
        """Periodic GUI update"""
        # Update every 100ms
        self.root.after(100, self.update_gui)
    
    def run(self):
        """Run the application"""
        self.root.mainloop()


# ================================================================================
# COMMAND LINE INTERFACE
# ================================================================================

class CommandLineInterface:
    """Command line version of the application"""
    
    def __init__(self):
        self.engine = ObjectDetectionEngine()
        self.running = False
        
    def run(self):
        """Run CLI version"""
        self.show_banner()
        
        while True:
            self.show_menu()
            choice = input("\nEnter your choice: ").strip()
            
            if choice == '1':
                self.webcam_mode()
            elif choice == '2':
                self.image_mode()
            elif choice == '3':
                self.video_mode()
            elif choice == '4':
                self.change_model()
            elif choice == '5':
                self.show_stats()
            elif choice == '6':
                print("\n👋 Thank you for using Ultimate Object Detection System!")
                break
            else:
                print("❌ Invalid choice!")
    
    def show_banner(self):
        """Show welcome banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════════╗
║              ULTIMATE OBJECT DETECTION SYSTEM v2.0               ║
║                       Command Line Interface                      ║
╚═══════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def show_menu(self):
        """Show main menu"""
        menu = """
╔═══════════════════════════════════════════════════════════════════╗
║                           MAIN MENU                               ║
╠═══════════════════════════════════════════════════════════════════╣
║  1. Webcam Detection (Real-time)                                 ║
║  2. Image File Detection                                          ║
║  3. Video File Detection                                          ║
║  4. Change Model                                                  ║
║  5. View Statistics                                               ║
║  6. Exit                                                          ║
╚═══════════════════════════════════════════════════════════════════╝
        """
        print(menu)
    
    def webcam_mode(self):
        """Webcam detection mode"""
        print("\n🎥 Starting webcam detection...")
        print("Controls: 'q' to quit, 's' for screenshot, 'c' to change confidence")
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame, detections = self.engine.detect_objects(frame)
            frame = self.engine.draw_detections(frame, detections)
            frame = self.engine.draw_stats(frame)
            
            # Display
            cv2.imshow('Object Detection - Press "q" to quit', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = self.engine.save_screenshot(frame, detections)
                print(f"✓ Screenshot saved: {filename}")
            elif key == ord('c'):
                self.engine.confidence_threshold = (self.engine.confidence_threshold + 0.1) % 1.0
                print(f"✓ Confidence threshold: {self.engine.confidence_threshold:.2f}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def image_mode(self):
        """Image file detection mode"""
        filename = input("Enter image filename: ").strip()
        
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            return
        
        frame = cv2.imread(filename)
        frame, detections = self.engine.detect_objects(frame)
        frame = self.engine.draw_detections(frame, detections)
        frame = self.engine.draw_stats(frame)
        
        # Show image
        cv2.imshow('Object Detection Result', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Ask to save
        save = input("Save result? (y/n): ").strip().lower()
        if save == 'y':
            output = f"detected_{filename}"
            cv2.imwrite(output, frame)
            print(f"✓ Saved as: {output}")
    
    def video_mode(self):
        """Video file detection mode"""
        filename = input("Enter video filename: ").strip()
        
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            return
        
        cap = cv2.VideoCapture(filename)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Ask to save output
        save_video = input("Save output video? (y/n): ").strip().lower()
        
        if save_video == 'y':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output_detection.mp4', fourcc, fps, (width, height))
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {total_frames} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            frame, detections = self.engine.detect_objects(frame)
            frame = self.engine.draw_detections(frame, detections)
            frame = self.engine.draw_stats(frame)
            
            if save_video == 'y':
                out.write(frame)
            
            # Show progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}%", end='\r')
            
            # Show frame (optional)
            cv2.imshow('Processing Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if save_video == 'y':
            out.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Processing complete! Processed {frame_count} frames")
    
    def change_model(self):
        """Change YOLO model"""
        print("\nAvailable Models:")
        for key, model in Config.MODELS.items():
            print(f"  {key}")
        
        choice = input("Select model (1-4): ").strip()
        model_key = f"{choice}: {list(Config.MODELS.values())[int(choice)-1]}"
        model_file = Config.MODELS[model_key]
        
        if self.engine.load_model(model_file):
            print(f"✓ Model changed to {model_key}")
    
    def show_stats(self):
        """Show detection statistics"""
        print("\n" + self.engine.generate_report())


# ================================================================================
# MAIN ENTRY POINT
# ================================================================================

def main():
    """Main entry point"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║              ULTIMATE OBJECT DETECTION SYSTEM v2.0               ║
║                         by AI Assistant                          ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    print("Select interface:")
    print("1. 🖥️  Graphical User Interface (GUI)")
    print("2. 📟 Command Line Interface (CLI)")
    print("3. ❌ Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        app = ObjectDetectionApp()
        app.run()
    elif choice == '2':
        cli = CommandLineInterface()
        cli.run()
    elif choice == '3':
        print("Goodbye! 👋")
        sys.exit(0)
    else:
        print("Invalid choice! Please select 1, 2, or 3.")
        main()


if __name__ == "__main__":
    main