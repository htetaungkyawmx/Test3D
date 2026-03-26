"""
Face Recognition System - Complete Implementation
Author: Your Name
Date: 2024
"""

import cv2
import numpy as np
import os
import pickle
import time
import json
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import threading
import queue
from collections import defaultdict

# InsightFace imports
try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
except ImportError as e:
    print(f"Error importing insightface: {e}")
    print("Please run: pip install insightface")
    exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    """
    Complete Face Recognition System with:
    - Real-time face detection
    - Face recognition with InsightFace
    - FAISS vector database for fast search
    - Support for RTSP IP cameras
    - Face registration system
    """
    
    def __init__(self, 
                 camera_source: str = "0",
                 known_faces_dir: str = "known_faces",
                 database_file: str = "saved_data/face_database.pkl",
                 config_file: str = "saved_data/config.json"):
        """
        Initialize the Face Recognition System
        
        Args:
            camera_source: Camera ID (0,1,2) or RTSP URL
            known_faces_dir: Directory for known face images
            database_file: Path to save face embeddings database
            config_file: Path to save configuration
        """
        self.camera_source = camera_source
        self.known_faces_dir = known_faces_dir
        self.database_file = database_file
        self.config_file = config_file
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize face analysis app
        self._init_face_models()
        
        # Known faces database
        self.known_face_embeddings = []
        self.known_face_names = []
        self.known_face_metadata = []
        
        # FAISS index for fast similarity search
        self.index = None
        
        # Recognition parameters
        self.similarity_threshold = self.config.get('similarity_threshold', 0.4)
        self.detection_threshold = self.config.get('detection_threshold', 0.5)
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.processing_times = []
        
        # Frame queue for threaded processing
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.running = False
        
        # Face tracking
        self.face_trackers = {}
        self.tracker_timeout = 30  # frames
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_faces_detected': 0,
            'total_recognitions': 0,
            'unknown_faces': 0,
            'start_time': time.time()
        }
        
        # Create necessary directories
        self._create_directories()
        
        # Load existing database
        self.load_database()
        
        # Scan known faces directory
        self.scan_known_faces()
        
        logger.info("Face Recognition System initialized successfully")
        
    def _init_face_models(self):
        """Initialize face detection and recognition models"""
        try:
            # Face analysis app for detection and recognition
            # Use GPU if available (ctx_id=0 for GPU, -1 for CPU)
            ctx_id = 0 if self.config.get('use_gpu', True) else -1
            
            self.app = FaceAnalysis(
                name='buffalo_l',  # buffalo_l is high accuracy, buffalo_m is faster
                root='saved_data',
                allowed_modules=['detection', 'recognition', 'landmark_2d_106']
            )
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            
            logger.info(f"Face models loaded successfully (GPU: {ctx_id == 0})")
            
        except Exception as e:
            logger.error(f"Failed to load face models: {e}")
            raise
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            'known_faces',
            'saved_data',
            'logs',
            'captured_faces',
            'backups'
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    
    def load_config(self) -> dict:
        """Load configuration from file"""
        default_config = {
            'similarity_threshold': 0.4,
            'detection_threshold': 0.5,
            'use_gpu': True,
            'camera_width': 1280,
            'camera_height': 720,
            'camera_fps': 30,
            'save_unknown_faces': True,
            'auto_backup': True,
            'recognition_interval': 10,  # frames between recognition
            'max_face_size': 640,
            'min_face_size': 64,
            'enable_tracking': True,
            'confidence_threshold': 0.7
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    default_config.update(config)
                    logger.info("Configuration loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        
        return default_config
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def scan_known_faces(self):
        """Scan the known faces directory and add faces to database"""
        if not os.path.exists(self.known_faces_dir):
            logger.warning(f"Known faces directory not found: {self.known_faces_dir}")
            return
        
        face_files = [f for f in os.listdir(self.known_faces_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not face_files:
            logger.info("No face images found in known_faces directory")
            return
        
        logger.info(f"Found {len(face_files)} face images to process")
        
        for filename in face_files:
            # Extract name from filename
            name = os.path.splitext(filename)[0]
            
            # Skip if already in database
            if name in self.known_face_names:
                continue
            
            # Load and process face image
            img_path = os.path.join(self.known_faces_dir, filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                # Detect faces in the image
                faces = self.app.get(img)
                
                if len(faces) > 0:
                    # Use the first detected face
                    embedding = faces[0].embedding
                    self.known_face_embeddings.append(embedding)
                    self.known_face_names.append(name)
                    
                    # Add metadata
                    metadata = {
                        'name': name,
                        'file': filename,
                        'added_date': datetime.now().isoformat(),
                        'num_samples': 1
                    }
                    self.known_face_metadata.append(metadata)
                    
                    logger.info(f"Added face: {name}")
                else:
                    logger.warning(f"No face detected in: {filename}")
            else:
                logger.error(f"Failed to load image: {filename}")
        
        # Update FAISS index
        if self.known_face_embeddings:
            self.build_faiss_index()
            self.save_database()
            logger.info(f"Database updated with {len(self.known_face_names)} faces")
    
    def build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        try:
            import faiss
            
            if len(self.known_face_embeddings) > 0:
                embeddings = np.array(self.known_face_embeddings).astype('float32')
                dimension = embeddings.shape[1]
                
                # Use IndexFlatIP for cosine similarity
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(embeddings)
                
                logger.info(f"FAISS index built with {len(embeddings)} embeddings")
            else:
                self.index = None
                
        except ImportError:
            logger.warning("FAISS not installed, using brute force search")
            self.index = None
    
    def save_database(self):
        """Save face database to file"""
        try:
            data = {
                'embeddings': self.known_face_embeddings,
                'names': self.known_face_names,
                'metadata': self.known_face_metadata,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.database_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Database saved to {self.database_file}")
            
            # Auto backup if enabled
            if self.config.get('auto_backup'):
                backup_file = f"backups/face_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                with open(backup_file, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"Backup saved to {backup_file}")
                
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
    
    def load_database(self):
        """Load face database from file"""
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'rb') as f:
                    data = pickle.load(f)
                
                self.known_face_embeddings = data.get('embeddings', [])
                self.known_face_names = data.get('names', [])
                self.known_face_metadata = data.get('metadata', [])
                
                logger.info(f"Loaded {len(self.known_face_names)} faces from database")
                
                if self.known_face_embeddings:
                    self.build_faiss_index()
                    
            except Exception as e:
                logger.error(f"Failed to load database: {e}")
    
    def recognize_face(self, face_embedding: np.ndarray) -> Tuple[str, float, int]:
        """
        Recognize a face by comparing with known faces
        
        Returns:
            name: Recognized name or "Unknown"
            confidence: Similarity score (0-1)
            index: Index of matched face (-1 if unknown)
        """
        if self.index is None or len(self.known_face_embeddings) == 0:
            return "Unknown", 0.0, -1
        
        # Convert to float32 and reshape
        embedding = np.array(face_embedding).astype('float32').reshape(1, -1)
        
        # Search in FAISS index
        similarities, indices = self.index.search(embedding, 1)
        
        # Get similarity score
        similarity = similarities[0][0]
        
        # FAISS IndexFlatIP returns cosine similarity (higher is better)
        # Convert to confidence score
        confidence = (similarity + 1) / 2  # Normalize to [0, 1]
        
        if confidence > self.similarity_threshold:
            name = self.known_face_names[indices[0][0]]
            return name, confidence, indices[0][0]
        else:
            self.stats['unknown_faces'] += 1
            return "Unknown", confidence, -1
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame: detect and recognize faces
        
        Returns:
            annotated_frame: Frame with bounding boxes and labels
            face_results: List of detection results
        """
        results = []
        
        # Face detection
        faces = self.app.get(frame)
        self.stats['total_faces_detected'] += len(faces)
        
        # Process each detected face
        for face in faces:
            # Extract embedding
            embedding = face.embedding
            
            # Recognize face
            name, confidence, idx = self.recognize_face(embedding)
            
            # Get bounding box
            bbox = face.bbox.astype(int)
            
            # Get landmarks
            landmarks = face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else None
            
            # Prepare result
            result = {
                'bbox': bbox,
                'name': name,
                'confidence': confidence,
                'landmarks': landmarks,
                'embedding': embedding if name == "Unknown" else None
            }
            results.append(result)
            
            if name != "Unknown":
                self.stats['total_recognitions'] += 1
            
            # Draw on frame
            frame = self.draw_face_result(frame, result)
        
        return frame, results
    
    def draw_face_result(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw bounding box and label for a face"""
        bbox = result['bbox']
        name = result['name']
        confidence = result['confidence']
        
        # Color based on recognition
        if name != "Unknown":
            color = (0, 255, 0)  # Green for known
            label = f"{name}: {confidence:.2f}"
        else:
            color = (0, 0, 255)  # Red for unknown
            label = f"Unknown: {confidence:.2f}"
        
        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (bbox[0], bbox[1] - 25), 
                     (bbox[0] + label_size[0], bbox[1]), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (bbox[0], bbox[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_statistics(self, frame: np.ndarray) -> np.ndarray:
        """Draw performance statistics on frame"""
        # Calculate FPS
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
        
        # Statistics text
        stats_text = [
            f"FPS: {self.fps:.1f}",
            f"Faces: {len(self.known_face_names)} known",
            f"Detected: {self.stats['total_faces_detected']}",
            f"Recognized: {self.stats['total_recognitions']}"
        ]
        
        # Draw stats on frame
        y_offset = 30
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw instructions
        instructions = [
            "Press 'q' - Quit",
            "Press 'a' - Add face",
            "Press 's' - Save snapshot",
            "Press 't' - Toggle stats"
        ]
        
        y_offset = frame.shape[0] - 100
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def add_face_interactive(self, frame: np.ndarray, faces: List) -> bool:
        """
        Interactive face addition to database
        """
        if len(faces) == 0:
            logger.warning("No face detected to add")
            return False
        
        # Use the first face
        face = faces[0]
        
        # Create a temporary window for name input
        name = input("Enter name for this face: ").strip()
        
        if not name:
            logger.warning("No name provided")
            return False
        
        # Check if name already exists
        if name in self.known_face_names:
            overwrite = input(f"Name '{name}' already exists. Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                return False
            
            # Remove existing entries
            indices = [i for i, n in enumerate(self.known_face_names) if n == name]
            for idx in sorted(indices, reverse=True):
                del self.known_face_names[idx]
                del self.known_face_embeddings[idx]
        
        # Add new face
        embedding = face.embedding
        self.known_face_embeddings.append(embedding)
        self.known_face_names.append(name)
        
        # Save face image
        bbox = face.bbox.astype(int)
        face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        face_img_path = f"captured_faces/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(face_img_path, face_img)
        
        # Update metadata
        metadata = {
            'name': name,
            'file': face_img_path,
            'added_date': datetime.now().isoformat(),
            'capture_method': 'interactive'
        }
        self.known_face_metadata.append(metadata)
        
        # Update FAISS index and save
        self.build_faiss_index()
        self.save_database()
        
        logger.info(f"Added new face: {name}")
        return True
    
    def save_snapshot(self, frame: np.ndarray, faces: List):
        """Save current frame with detection results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save full frame
        frame_path = f"captured_faces/frame_{timestamp}.jpg"
        cv2.imwrite(frame_path, frame)
        
        # Save individual faces
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            name, confidence, _ = self.recognize_face(face.embedding)
            face_path = f"captured_faces/face_{timestamp}_{i}_{name}.jpg"
            cv2.imwrite(face_path, face_img)
        
        logger.info(f"Snapshot saved: {frame_path}")
    
    def camera_thread(self):
        """Thread for capturing frames from camera"""
        cap = cv2.VideoCapture(self.camera_source)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera_height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['camera_fps'])
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera: {self.camera_source}")
            self.running = False
            return
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                continue
            
            # Put frame in queue (non-blocking)
            try:
                self.frame_queue.put(frame, timeout=1)
            except queue.Full:
                continue
        
        cap.release()
    
    def processing_thread(self):
        """Thread for processing frames"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                
                # Process frame
                start_time = time.time()
                processed_frame, results = self.process_frame(frame)
                processing_time = time.time() - start_time
                
                # Track processing times
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
                
                # Draw statistics
                show_stats = self.config.get('show_stats', True)
                if show_stats:
                    processed_frame = self.draw_statistics(processed_frame)
                
                # Put result in queue
                try:
                    self.result_queue.put((processed_frame, results), timeout=1)
                except queue.Full:
                    continue
                    
                self.frame_count += 1
                self.stats['total_frames'] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
    
    def run(self):
        """
        Main function to run the face recognition system
        """
        logger.info("Starting Face Recognition System...")
        logger.info(f"Camera source: {self.camera_source}")
        logger.info(f"Database contains {len(self.known_face_names)} faces")
        
        # Start threads
        self.running = True
        camera_thread = threading.Thread(target=self.camera_thread)
        processing_thread = threading.Thread(target=self.processing_thread)
        
        camera_thread.start()
        processing_thread.start()
        
        # Variables for interactive features
        show_stats = True
        last_stats_time = time.time()
        
        try:
            while self.running:
                # Get processed frame
                try:
                    frame, results = self.result_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Show frame
                cv2.imshow("Face Recognition System", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Quit signal received")
                    break
                    
                elif key == ord('a'):
                    # Add new face
                    faces = self.app.get(frame)
                    if len(faces) > 0:
                        self.add_face_interactive(frame, faces)
                    else:
                        logger.warning("No face detected to add")
                        
                elif key == ord('s'):
                    # Save snapshot
                    faces = self.app.get(frame)
                    self.save_snapshot(frame, faces)
                    
                elif key == ord('t'):
                    # Toggle statistics display
                    show_stats = not show_stats
                    self.config['show_stats'] = show_stats
                    logger.info(f"Statistics display: {show_stats}")
                    
                elif key == ord('c'):
                    # Clear unknown faces from database (if any)
                    logger.info("Clear unknown faces feature - not implemented")
                
                # Print statistics every 10 seconds
                if time.time() - last_stats_time > 10:
                    avg_processing = np.mean(self.processing_times) if self.processing_times else 0
                    logger.info(f"Stats - FPS: {self.fps:.1f}, "
                              f"Avg Processing: {avg_processing*1000:.1f}ms, "
                              f"Known Faces: {len(self.known_face_names)}")
                    last_stats_time = time.time()
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            
        finally:
            # Cleanup
            self.running = False
            camera_thread.join(timeout=2)
            processing_thread.join(timeout=2)
            cv2.destroyAllWindows()
            
            # Print final statistics
            runtime = time.time() - self.stats['start_time']
            logger.info("=" * 50)
            logger.info("Final Statistics:")
            logger.info(f"Total runtime: {runtime:.1f} seconds")
            logger.info(f"Total frames: {self.stats['total_frames']}")
            logger.info(f"Faces detected: {self.stats['total_faces_detected']}")
            logger.info(f"Faces recognized: {self.stats['total_recognitions']}")
            logger.info(f"Unknown faces: {self.stats['unknown_faces']}")
            logger.info(f"Database size: {len(self.known_face_names)} faces")
            logger.info("=" * 50)
            
            self.save_config()
            logger.info("System stopped successfully")


class FaceRecognitionCLI:
    """
    Command Line Interface for Face Recognition System
    """
    
    def __init__(self):
        self.system = None
        
    def display_banner(self):
        """Display welcome banner"""
        print("=" * 60)
        print("  FACE RECOGNITION SYSTEM v1.0")
        print("  with AI Detection & FAISS Database")
        print("=" * 60)
        print("\nDeveloper: Your Name")
        print("Requirements: Python 3.8+, GPU Recommended")
        print("=" * 60)
    
    def get_camera_source(self):
        """Get camera source from user"""
        print("\nSelect camera source:")
        print("1. Built-in webcam (0)")
        print("2. USB Camera (1)")
        print("3. IP Camera (RTSP URL)")
        print("4. Video File")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            return "0"
        elif choice == '2':
            return "1"
        elif choice == '3':
            rtsp_url = input("Enter RTSP URL: ").strip()
            return rtsp_url
        elif choice == '4':
            video_path = input("Enter video file path: ").strip()
            return video_path
        else:
            print("Invalid choice, using default webcam")
            return "0"
    
    def setup_known_faces(self):
        """Guide user to set up known faces"""
        print("\n" + "=" * 60)
        print("SETUP KNOWN FACES")
        print("=" * 60)
        print("\nTo add known faces:")
        print("1. Place face images in 'known_faces' folder")
        print("2. Filename should be the person's name (e.g., 'john_doe.jpg')")
        print("3. Make sure each image contains a clear front-facing face")
        print("4. Run the system - faces will be automatically loaded")
        print("\nOr you can add faces interactively while the system is running")
        print("by pressing 'a' key when a face is detected.")
        
        input("\nPress Enter to continue...")
    
    def configure_system(self):
        """Interactive configuration"""
        print("\n" + "=" * 60)
        print("SYSTEM CONFIGURATION")
        print("=" * 60)
        
        config = {}
        
        # Similarity threshold
        print("\nSimilarity threshold (0.3-0.7):")
        print("  Lower = more matches (but more false positives)")
        print("  Higher = stricter matching (but may miss known faces)")
        config['similarity_threshold'] = float(input("Enter threshold [default 0.4]: ").strip() or "0.4")
        
        # Use GPU
        use_gpu = input("\nUse GPU if available? (y/n) [default y]: ").strip().lower()
        config['use_gpu'] = use_gpu != 'n'
        
        # Camera resolution
        print("\nCamera resolution:")
        config['camera_width'] = int(input("Width [default 1280]: ").strip() or "1280")
        config['camera_height'] = int(input("Height [default 720]: ").strip() or "720")
        
        return config
    
    def run(self):
        """Main CLI entry point"""
        self.display_banner()
        
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            print("Error: Python 3.8 or higher required")
            return
        
        # Setup
        self.setup_known_faces()
        
        # Get camera source
        camera_source = self.get_camera_source()
        
        # Configure system
        config = self.configure_system()
        
        # Create and run system
        print("\nInitializing system...")
        
        try:
            self.system = FaceRecognitionSystem(
                camera_source=camera_source,
                config_file="saved_data/config.json"
            )
            
            # Update config
            for key, value in config.items():
                self.system.config[key] = value
            self.system.save_config()
            
            print("\nSystem ready!")
            print("\nControls:")
            print("  'q' - Quit")
            print("  'a' - Add current face to database")
            print("  's' - Save snapshot")
            print("  't' - Toggle statistics display")
            print("\nStarting in 3 seconds...")
            
            import time
            time.sleep(3)
            
            # Run system
            self.system.run()
            
        except Exception as e:
            print(f"\nError: {e}")
            print("\nTroubleshooting:")
            print("1. Check if camera is connected")
            print("2. Verify RTSP URL if using IP camera")
            print("3. Ensure all dependencies are installed")
            print("4. Check GPU drivers if using GPU")


if __name__ == "__main__":
    # Run CLI
    cli = FaceRecognitionCLI()
    cli.run()