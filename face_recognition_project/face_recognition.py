#!/usr/bin/env python3
"""
ULTIMATE FACE RECOGNITION SYSTEM
- High Accuracy Detection
- Auto Save Unknown Faces
- Auto Save Matched Faces
- Real-time Statistics
- Easy to Use
"""

import cv2
import numpy as np
import os
import pickle
import time
import json
import shutil
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import threading
import queue
from collections import defaultdict

# InsightFace imports
try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    print("❌ Please install insightface: pip install insightface")
    exit(1)

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ultimate_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UltimateFaceRecognition:
    """
    Ultimate Face Recognition System
    Features:
    - Unknown face auto-save
    - Matched face auto-save
    - High accuracy detection
    - Real-time statistics
    - Easy controls
    """
    
    def __init__(self):
        """Initialize the system"""
        
        # Directories
        self.known_faces_dir = "known_faces"
        self.unknown_faces_dir = "unknown_faces"
        self.matched_faces_dir = "matched_faces"
        self.database_file = "saved_data/ultimate_database.pkl"
        self.config_file = "saved_data/ultimate_config.json"
        
        # Create directories
        self._create_directories()
        
        # Load config
        self.config = self._load_config()
        
        # Initialize face models
        self._init_face_models()
        
        # Databases
        self.known_embeddings = []
        self.known_names = []
        self.known_metadata = []
        
        # FAISS index
        self.index = None
        
        # Recognition settings
        self.threshold = self.config.get('threshold', 0.45)  # Lower = stricter
        self.save_unknown = self.config.get('save_unknown', True)
        self.save_matched = self.config.get('save_matched', True)
        self.save_interval = self.config.get('save_interval', 3)  # Seconds between saves
        
        # Tracking
        self.last_unknown_save = {}
        self.last_match_save = {}
        self.match_history = []
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'faces_detected': 0,
            'known_found': 0,
            'unknown_found': 0,
            'saved_unknown': 0,
            'saved_matches': 0,
            'start_time': time.time()
        }
        
        # Performance
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Threading
        self.running = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        
        # Load existing data
        self._load_database()
        self._scan_known_faces()
        
        logger.info("="*50)
        logger.info("ULTIMATE FACE RECOGNITION SYSTEM READY")
        logger.info(f"Known faces: {len(self.known_names)}")
        logger.info(f"Threshold: {self.threshold}")
        logger.info("="*50)
    
    def _create_directories(self):
        """Create all necessary directories"""
        dirs = [
            self.known_faces_dir,
            self.unknown_faces_dir,
            self.matched_faces_dir,
            'saved_data',
            'logs',
            'backups',
            'daily_saves'
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def _load_config(self) -> dict:
        """Load configuration"""
        default = {
            'threshold': 0.45,
            'save_unknown': True,
            'save_matched': True,
            'save_interval': 3,
            'detection_size': (640, 640),
            'camera_width': 1280,
            'camera_height': 720,
            'use_gpu': False,
            'show_stats': True,
            'min_face_size': 80
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    default.update(data)
            except:
                pass
        
        return default
    
    def _save_config(self):
        """Save configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except:
            pass
    
    def _init_face_models(self):
        """Initialize face models"""
        try:
            ctx_id = 0 if self.config.get('use_gpu', False) else -1
            
            self.app = FaceAnalysis(
                name='buffalo_l',
                root='saved_data',
                allowed_modules=['detection', 'recognition']
            )
            
            det_size = self.config.get('detection_size', (640, 640))
            self.app.prepare(ctx_id=ctx_id, det_size=det_size)
            
            logger.info(f"✓ Face models loaded (Detection size: {det_size})")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _scan_known_faces(self):
        """Scan known faces directory and add to database"""
        if not os.path.exists(self.known_faces_dir):
            return
        
        files = [f for f in os.listdir(self.known_faces_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in files:
            name = os.path.splitext(filename)[0]
            
            if name in self.known_names:
                continue
            
            img_path = os.path.join(self.known_faces_dir, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            faces = self.app.get(img)
            
            if len(faces) > 0:
                # Use best face
                best = max(faces, key=lambda x: x.det_score)
                embedding = best.embedding
                
                self.known_embeddings.append(embedding)
                self.known_names.append(name)
                self.known_metadata.append({
                    'name': name,
                    'file': filename,
                    'quality': best.det_score,
                    'added': datetime.now().isoformat()
                })
                
                logger.info(f"✓ Added: {name} (quality: {best.det_score:.2f})")
        
        if self.known_embeddings:
            self._build_index()
            self._save_database()
    
    def _build_index(self):
        """Build FAISS index"""
        try:
            import faiss
            
            embeddings = np.array(self.known_embeddings).astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
            
            logger.info(f"✓ FAISS index built with {len(embeddings)} faces")
            
        except ImportError:
            logger.warning("FAISS not available")
            self.index = None
    
    def _save_database(self):
        """Save database to file"""
        try:
            data = {
                'embeddings': self.known_embeddings,
                'names': self.known_names,
                'metadata': self.known_metadata,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.database_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Daily backup
            date = datetime.now().strftime('%Y%m%d')
            backup_file = f"backups/database_{date}.pkl"
            shutil.copy2(self.database_file, backup_file)
            
        except Exception as e:
            logger.error(f"Failed to save: {e}")
    
    def _load_database(self):
        """Load database from file"""
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'rb') as f:
                    data = pickle.load(f)
                
                self.known_embeddings = data.get('embeddings', [])
                self.known_names = data.get('names', [])
                self.known_metadata = data.get('metadata', [])
                
                logger.info(f"✓ Loaded {len(self.known_names)} faces")
                
                if self.known_embeddings:
                    self._build_index()
                    
            except Exception as e:
                logger.error(f"Failed to load: {e}")
    
    def _recognize(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Recognize face
        
        Returns:
            name: Person name or "Unknown"
            confidence: Confidence score (0-1)
        """
        if self.index is None or len(self.known_names) == 0:
            return "Unknown", 0.0
        
        try:
            import faiss
            
            emb = np.array(embedding).astype('float32').reshape(1, -1)
            faiss.normalize_L2(emb)
            
            similarities, indices = self.index.search(emb, 1)
            similarity = similarities[0][0]
            
            # Convert to confidence (0-1)
            confidence = (similarity + 1) / 2
            
            if confidence > self.threshold:
                name = self.known_names[indices[0][0]]
                return name, confidence
            else:
                return "Unknown", confidence
                
        except:
            return "Unknown", 0.0
    
    def _save_unknown(self, face_img: np.ndarray, frame: np.ndarray, 
                      confidence: float) -> Optional[str]:
        """Save unknown face"""
        if not self.save_unknown:
            return None
        
        # Check interval
        current_time = time.time()
        if current_time - self.last_unknown_save.get('time', 0) < self.save_interval:
            return None
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        filename = f"unknown_{timestamp}_conf{confidence:.2f}.jpg"
        filepath = os.path.join(self.unknown_faces_dir, filename)
        
        # Save face
        cv2.imwrite(filepath, face_img)
        
        # Save full frame
        frame_path = os.path.join(self.unknown_faces_dir, f"frame_{timestamp}.jpg")
        cv2.imwrite(frame_path, frame)
        
        # Update tracking
        self.last_unknown_save = {'time': current_time, 'file': filename}
        self.stats['saved_unknown'] += 1
        
        logger.info(f"📸 UNKNOWN SAVED: {filename}")
        return filepath
    
    def _save_match(self, face_img: np.ndarray, frame: np.ndarray,
                    name: str, confidence: float) -> Optional[str]:
        """Save matched face"""
        if not self.save_matched:
            return None
        
        # Check interval for this person
        current_time = time.time()
        last_time = self.last_match_save.get(name, 0)
        if current_time - last_time < self.save_interval:
            return None
        
        # Create person directory
        person_dir = os.path.join(self.matched_faces_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        filename = f"{name}_{timestamp}_conf{confidence:.2f}.jpg"
        filepath = os.path.join(person_dir, filename)
        
        # Save face
        cv2.imwrite(filepath, face_img)
        
        # Also save to daily folder
        daily_dir = f"daily_saves/{datetime.now().strftime('%Y%m%d')}/{name}"
        os.makedirs(daily_dir, exist_ok=True)
        daily_path = os.path.join(daily_dir, filename)
        cv2.imwrite(daily_path, face_img)
        
        # Record match
        record = {
            'name': name,
            'confidence': confidence,
            'timestamp': timestamp,
            'file': filepath
        }
        self.match_history.append(record)
        
        # Keep last 500 matches
        if len(self.match_history) > 500:
            self.match_history = self.match_history[-500:]
        
        # Update tracking
        self.last_match_save[name] = current_time
        self.stats['saved_matches'] += 1
        
        logger.info(f"✓ MATCH SAVED: {name} (conf: {confidence:.2f})")
        return filepath
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
        """Process a single frame"""
        results = []
        
        # Detect faces
        faces = self.app.get(frame)
        self.stats['faces_detected'] += len(faces)
        
        for face in faces:
            # Get bounding box
            bbox = face.bbox.astype(int)
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            face_size = min(face_width, face_height)
            
            # Skip small faces
            if face_size < self.config.get('min_face_size', 80):
                continue
            
            # Recognize
            name, confidence = self._recognize(face.embedding)
            
            # Extract face image
            face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            # Handle based on recognition
            if name != "Unknown" and confidence > 0.5:
                self.stats['known_found'] += 1
                self._save_match(face_img, frame, name, confidence)
            else:
                self.stats['unknown_found'] += 1
                self._save_unknown(face_img, frame, confidence)
            
            # Prepare result
            result = {
                'bbox': bbox,
                'name': name,
                'confidence': confidence,
                'face_size': face_size
            }
            results.append(result)
            
            # Draw on frame
            frame = self._draw_face(frame, result)
        
        return frame, results
    
    def _draw_face(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw face detection result"""
        bbox = result['bbox']
        name = result['name']
        confidence = result['confidence']
        
        # Choose color
        if name != "Unknown":
            if confidence > 0.8:
                color = (0, 255, 0)      # Green - High confidence
            elif confidence > 0.6:
                color = (0, 255, 255)    # Yellow - Medium
            else:
                color = (0, 165, 255)    # Orange - Low
            label = f"{name} [{confidence:.2f}]"
        else:
            color = (0, 0, 255)          # Red - Unknown
            label = f"Unknown [{confidence:.2f}]"
        
        # Draw box
        thickness = 2 if confidence > 0.6 else 1
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
        
        # Draw label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (bbox[0], bbox[1] - 25), 
                     (bbox[0] + label_size[0], bbox[1]), color, -1)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw confidence bar
        bar_width = int((bbox[2] - bbox[0]) * confidence)
        cv2.rectangle(frame, (bbox[0], bbox[3] + 5), 
                     (bbox[0] + bar_width, bbox[3] + 12), color, -1)
        
        return frame
    
    def _draw_stats(self, frame: np.ndarray) -> np.ndarray:
        """Draw statistics panel"""
        # Calculate FPS
        self.frame_count += 1
        if time.time() - self.last_fps_time > 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = time.time()
        
        # Stats text
        stats = [
            f"FPS: {self.fps}",
            f"Known: {len(self.known_names)}",
            f"Detected: {self.stats['faces_detected']}",
            f"Known Found: {self.stats['known_found']}",
            f"Unknown Found: {self.stats['unknown_found']}",
            f"Saved Unknown: {self.stats['saved_unknown']}",
            f"Saved Matches: {self.stats['saved_matches']}",
            f"Threshold: {self.threshold:.2f}"
        ]
        
        # Draw stats
        y = 30
        for text in stats:
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
            y += 25
        
        # Draw controls
        controls = [
            "q: Quit | a: Add Face | +: Higher threshold",
            "-: Lower threshold | s: Screenshot | c: Clear stats"
        ]
        
        y = frame.shape[0] - 50
        for text in controls:
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1)
            y += 25
        
        return frame
    
    def add_face(self, frame: np.ndarray, faces: List) -> bool:
        """Add new face to database"""
        if len(faces) == 0:
            print("\n❌ No face detected! Please look at camera.")
            return False
        
        # Get best face
        best = max(faces, key=lambda x: x.det_score)
        bbox = best.bbox.astype(int)
        face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        # Show preview
        print("\n" + "="*50)
        print("ADD NEW FACE")
        print("="*50)
        
        # Show preview window
        preview = cv2.resize(face_img, (200, 200))
        cv2.imshow("Preview - Press ENTER to continue", preview)
        cv2.waitKey(1)
        
        name = input("\nEnter name: ").strip()
        cv2.destroyWindow("Preview - Press ENTER to continue")
        
        if not name:
            print("❌ No name provided")
            return False
        
        # Check quality
        if best.det_score < 0.5:
            print(f"⚠ Low quality face (score: {best.det_score:.2f})")
        
        # Add to database
        self.known_embeddings.append(best.embedding)
        self.known_names.append(name)
        
        # Save image
        img_path = os.path.join(self.known_faces_dir, f"{name}.jpg")
        cv2.imwrite(img_path, face_img)
        
        # Save metadata
        self.known_metadata.append({
            'name': name,
            'file': f"{name}.jpg",
            'quality': best.det_score,
            'added': datetime.now().isoformat()
        })
        
        # Rebuild index
        self._build_index()
        self._save_database()
        
        print(f"\n✅ Face '{name}' added successfully!")
        print(f"   Quality: {best.det_score:.2f}")
        
        return True
    
    def camera_thread(self):
        """Camera capture thread"""
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera_width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera_height'])
            
            if not cap.isOpened():
                logger.error("Failed to open camera!")
                self.running = False
                return
            
            logger.info("✓ Camera opened successfully")
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                try:
                    self.frame_queue.put(frame, timeout=1)
                except queue.Full:
                    continue
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Camera error: {e}")
            self.running = False
    
    def processing_thread(self):
        """Frame processing thread"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                
                # Process
                processed, results = self.process_frame(frame)
                
                # Add stats
                if self.config.get('show_stats', True):
                    processed = self._draw_stats(processed)
                
                # Queue result
                try:
                    self.result_queue.put(processed, timeout=1)
                except queue.Full:
                    continue
                
                self.stats['total_frames'] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
    
    def run(self):
        """Main run loop"""
        logger.info("Starting system...")
        
        # Start threads
        self.running = True
        cam_thread = threading.Thread(target=self.camera_thread)
        proc_thread = threading.Thread(target=self.processing_thread)
        
        cam_thread.start()
        proc_thread.start()
        
        print("\n" + "="*60)
        print("SYSTEM READY!")
        print("="*60)
        print("\nCONTROLS:")
        print("  q - Quit")
        print("  a - Add new face to database")
        print("  s - Save screenshot")
        print("  + - Increase threshold (stricter)")
        print("  - - Decrease threshold (looser)")
        print("  c - Clear statistics display")
        print("  t - Toggle statistics")
        print("\n" + "="*60)
        print("Waiting for faces...")
        print("="*60 + "\n")
        
        try:
            while self.running:
                try:
                    frame = self.result_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                cv2.imshow("Ultimate Face Recognition", frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                    
                elif key == ord('a'):
                    faces = self.app.get(frame)
                    self.add_face(frame, faces)
                    
                elif key == ord('s'):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    cv2.imwrite(f"screenshot_{timestamp}.jpg", frame)
                    print(f"\n📸 Screenshot saved: screenshot_{timestamp}.jpg")
                    
                elif key == ord('+') or key == ord('='):
                    self.threshold = min(0.7, self.threshold + 0.02)
                    self.config['threshold'] = self.threshold
                    print(f"\n🔧 Threshold: {self.threshold:.2f} (Stricter)")
                    
                elif key == ord('-') or key == ord('_'):
                    self.threshold = max(0.3, self.threshold - 0.02)
                    self.config['threshold'] = self.threshold
                    print(f"\n🔧 Threshold: {self.threshold:.2f} (Looser)")
                    
                elif key == ord('t'):
                    self.config['show_stats'] = not self.config.get('show_stats', True)
                    print(f"\n📊 Statistics display: {self.config['show_stats']}")
                    
                elif key == ord('c'):
                    self.stats = {k: 0 for k in self.stats}
                    self.stats['start_time'] = time.time()
                    print("\n🗑 Statistics cleared")
                    
        except KeyboardInterrupt:
            pass
        
        finally:
            self.running = False
            cam_thread.join(timeout=2)
            proc_thread.join(timeout=2)
            cv2.destroyAllWindows()
            
            # Final report
            runtime = time.time() - self.stats['start_time']
            print("\n" + "="*60)
            print("FINAL REPORT")
            print("="*60)
            print(f"Runtime: {runtime:.1f} seconds")
            print(f"Frames: {self.stats['total_frames']}")
            print(f"Faces Detected: {self.stats['faces_detected']}")
            print(f"Known Faces Found: {self.stats['known_found']}")
            print(f"Unknown Faces Found: {self.stats['unknown_found']}")
            print(f"Unknown Faces Saved: {self.stats['saved_unknown']}")
            print(f"Matches Saved: {self.stats['saved_matches']}")
            print(f"Database Size: {len(self.known_names)} faces")
            print("="*60)
            
            self._save_config()
            logger.info("System stopped")


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("  ULTIMATE FACE RECOGNITION SYSTEM")
    print("  Auto Save Unknown & Matched Faces")
    print("="*60)
    
    # Check camera
    print("\n🔍 Checking camera...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("✅ Camera working!")
        cap.release()
    else:
        print("❌ Camera not found! Please check:")
        print("   1. Camera connected?")
        print("   2. System Settings > Privacy > Camera")
        print("   3. Terminal allowed to use camera?")
        return
    
    # Ask for mode
    print("\nSelect mode:")
    print("  1. High Accuracy (slower, best for important cases)")
    print("  2. Balanced (recommended for daily use)")
    print("  3. Fast (faster, lower accuracy)")
    
    mode = input("\nEnter choice (1-3) [default 2]: ").strip()
    
    # Create system
    system = UltimateFaceRecognition()
    
    # Configure based on mode
    if mode == '1':
        system.config['detection_size'] = (800, 800)
        system.threshold = 0.42
        system.config['threshold'] = 0.42
        print("\n⚡ Mode: High Accuracy")
        
    elif mode == '3':
        system.config['detection_size'] = (320, 320)
        system.threshold = 0.48
        system.config['threshold'] = 0.48
        print("\n⚡ Mode: Fast")
        
    else:
        system.config['detection_size'] = (640, 640)
        system.threshold = 0.45
        system.config['threshold'] = 0.45
        print("\n⚡ Mode: Balanced")
    
    # Save config
    system._save_config()
    
    print(f"   Detection size: {system.config['detection_size']}")
    print(f"   Threshold: {system.threshold}")
    print(f"   Save unknown: {system.save_unknown}")
    print(f"   Save matches: {system.save_matched}")
    
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    # Run system
    system.run()


if __name__ == "__main__":
    main()