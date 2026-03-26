#!/usr/bin/env python3
"""
IPHONE FACE ID STYLE FACE LOCK SYSTEM
- iPhone-like Face ID animation
- Head movement detection (look left, right, up, down)
- Real-time authentication
- Success message with animation
- Multiple angle registration
"""

import cv2
import numpy as np
import os
import pickle
import time
import json
import math
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import threading
import queue

# InsightFace imports
try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    print("❌ Please install insightface: pip install insightface")
    exit(1)

# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IPhoneFaceID:
    """
    iPhone Style Face ID System
    Features:
    - Head movement detection (left, right, up, down)
    - Multi-angle face registration
    - Smooth animations
    - Success/failure feedback
    """
    
    def __init__(self):
        # Directories
        self.users_dir = "iphone_face_users"
        self.database_file = "saved_data/iphone_face_database.pkl"
        
        # Create directories
        os.makedirs(self.users_dir, exist_ok=True)
        os.makedirs("saved_data", exist_ok=True)
        
        # Initialize face models
        self.app = FaceAnalysis(name='buffalo_l', root='saved_data')
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        
        # User database
        self.users = {}
        self.current_user = None
        
        # Face ID state
        self.scanning = False
        self.registered_faces = []
        self.registration_stage = 0
        self.registration_angles = [
            {"name": "Center", "angle": 0, "text": "Look straight"},
            {"name": "Left", "angle": -30, "text": "Turn head left"},
            {"name": "Right", "angle": 30, "text": "Turn head right"},
            {"name": "Up", "angle": 20, "text": "Look up"},
            {"name": "Down", "angle": -20, "text": "Look down"}
        ]
        
        # Authentication
        self.auth_threshold = 0.85
        self.auth_result = None
        self.auth_animation = 0
        
        # Load database
        self._load_database()
        
        logger.info("iPhone Face ID System Ready")
    
    def _load_database(self):
        """Load user database"""
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'rb') as f:
                    self.users = pickle.load(f)
                logger.info(f"Loaded {len(self.users)} users")
            except:
                pass
    
    def _save_database(self):
        """Save user database"""
        with open(self.database_file, 'wb') as f:
            pickle.dump(self.users, f)
    
    def _extract_face(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Extract face from frame"""
        faces = self.app.get(frame)
        if len(faces) == 0:
            return None, None
        
        best = max(faces, key=lambda x: x.det_score)
        embedding = best.embedding
        bbox = best.bbox.astype(int)
        
        # Calculate head pose (simplified)
        landmarks = best.landmark_2d_106 if hasattr(best, 'landmark_2d_106') else None
        head_angle = self._estimate_head_angle(landmarks) if landmarks is not None else 0
        
        metadata = {
            'bbox': bbox,
            'detection_score': best.det_score,
            'head_angle': head_angle,
            'face_size': min(bbox[2]-bbox[0], bbox[3]-bbox[1])
        }
        
        return embedding, metadata
    
    def _estimate_head_angle(self, landmarks) -> float:
        """Estimate head angle from landmarks"""
        try:
            # Use eye and nose landmarks
            left_eye = landmarks[0]  # Approximate
            right_eye = landmarks[1]
            nose = landmarks[2]
            
            # Calculate angle
            eye_center = (left_eye + right_eye) / 2
            dx = nose[0] - eye_center[0]
            dy = nose[1] - eye_center[1]
            
            angle = np.arctan2(dx, dy) * 180 / np.pi
            return max(-45, min(45, angle))
        except:
            return 0
    
    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        similarity = np.dot(emb1_norm, emb2_norm)
        return (similarity + 1) / 2
    
    def register_user(self, name: str) -> bool:
        """Register new user with multiple angles"""
        print(f"\n{'='*50}")
        print(f"📱 REGISTERING USER: {name}")
        print(f"{'='*50}")
        print("Please follow the instructions on screen...")
        print("Move your head as directed\n")
        
        self.scanning = True
        self.registration_stage = 0
        self.registered_faces = []
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ Camera not found!")
            return False
        
        start_time = time.time()
        
        while self.scanning and self.registration_stage < len(self.registration_angles):
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get current stage
            stage = self.registration_angles[self.registration_stage]
            angle_name = stage["name"]
            angle_text = stage["text"]
            target_angle = stage["angle"]
            
            # Extract face
            embedding, metadata = self._extract_face(frame)
            
            # Draw UI
            frame = self._draw_registration_ui(frame, self.registration_stage + 1, 
                                                len(self.registration_angles), 
                                                angle_text, metadata)
            
            # Check if face detected and angle matches
            if embedding is not None and metadata is not None:
                current_angle = metadata['head_angle']
                
                # Draw angle indicator
                angle_diff = abs(current_angle - target_angle) if target_angle != 0 else abs(current_angle)
                
                if angle_diff < 15:  # Angle within range
                    # Show capturing animation
                    progress = min(1.0, (time.time() - start_time) / 2)
                    frame = self._draw_capture_animation(frame, progress)
                    
                    if progress >= 1.0:
                        # Capture face
                        self.registered_faces.append({
                            'embedding': embedding,
                            'angle': current_angle,
                            'stage': self.registration_stage
                        })
                        
                        print(f"  ✓ Captured {angle_name} angle")
                        self.registration_stage += 1
                        start_time = time.time()
                        
                        if self.registration_stage < len(self.registration_angles):
                            # Show success message
                            frame = self._draw_success_message(frame, "Great! Next position...")
                            cv2.imshow("iPhone Face ID", frame)
                            cv2.waitKey(1000)
            
            cv2.imshow("iPhone Face ID", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.scanning = False
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(self.registered_faces) >= 3:  # At least 3 angles
            # Save user data
            avg_embedding = np.mean([f['embedding'] for f in self.registered_faces], axis=0)
            
            user_data = {
                'name': name,
                'embeddings': [f['embedding'] for f in self.registered_faces],
                'avg_embedding': avg_embedding,
                'angles': [f['angle'] for f in self.registered_faces],
                'registered_date': datetime.now().isoformat(),
                'access_count': 0
            }
            
            self.users[name] = user_data
            self._save_database()
            
            print(f"\n{'='*50}")
            print(f"✅ SUCCESS! User '{name}' registered successfully!")
            print(f"   Captured {len(self.registered_faces)} angles")
            print(f"{'='*50}\n")
            
            return True
        else:
            print(f"\n❌ Registration failed. Only captured {len(self.registered_faces)} angles")
            return False
    
    def _draw_registration_ui(self, frame: np.ndarray, current: int, total: int, 
                               instruction: str, metadata: dict) -> np.ndarray:
        """Draw registration UI"""
        h, w = frame.shape[:2]
        
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(frame, "FACE ID SETUP", (w//2 - 120, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Progress
        progress_width = 300
        progress_height = 8
        progress_x = w//2 - progress_width//2
        progress_y = 130
        filled = int(progress_width * current / total)
        
        cv2.rectangle(frame, (progress_x, progress_y), 
                     (progress_x + progress_width, progress_y + progress_height),
                     (100, 100, 100), -1)
        cv2.rectangle(frame, (progress_x, progress_y), 
                     (progress_x + filled, progress_y + progress_height),
                     (0, 255, 0), -1)
        
        # Instruction
        cv2.putText(frame, instruction, (w//2 - 150, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Step indicator
        cv2.putText(frame, f"Step {current} of {total}", (w//2 - 80, 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Face circle indicator
        if metadata is not None:
            bbox = metadata['bbox']
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            radius = (bbox[2] - bbox[0]) // 2
            
            # Draw circle
            cv2.circle(frame, (center_x, center_y), radius + 10, (0, 255, 0), 3)
        
        return frame
    
    def _draw_capture_animation(self, frame: np.ndarray, progress: float) -> np.ndarray:
        """Draw capture animation"""
        h, w = frame.shape[:2]
        
        # Create circular progress
        center = (w//2, h//2)
        radius = 100
        angle = int(360 * progress)
        
        # Draw circle
        cv2.ellipse(frame, center, (radius, radius), 0, 0, angle, (0, 255, 0), 10)
        
        return frame
    
    def _draw_success_message(self, frame: np.ndarray, message: str) -> np.ndarray:
        """Draw success message"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, message, (w//2 - 150, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def authenticate(self, frame: np.ndarray) -> Tuple[bool, str, float]:
        """Authenticate face"""
        embedding, metadata = self._extract_face(frame)
        
        if embedding is None:
            return False, "No Face", 0.0
        
        best_match = None
        best_score = 0
        best_name = "Unknown"
        
        for name, user in self.users.items():
            # Compare with stored embeddings
            for stored_emb in user['embeddings']:
                score = self._calculate_similarity(embedding, stored_emb)
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_match = user
        
        if best_score > self.auth_threshold:
            if best_match:
                best_match['access_count'] += 1
                self._save_database()
            return True, best_name, best_score
        else:
            return False, "Unknown", best_score
    
    def _draw_face_id_ui(self, frame: np.ndarray, success: bool, name: str, 
                          confidence: float) -> np.ndarray:
        """Draw iPhone Face ID style UI"""
        h, w = frame.shape[:2]
        
        # Create blur effect on top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Face ID icon
        center_x = w//2
        center_y = 75
        radius = 40
        
        if success:
            color = (0, 255, 0)
            status_text = "✅ Face ID"
            sub_text = f"Welcome back, {name}!"
            
            # Draw checkmark animation
            cv2.circle(frame, (center_x, center_y), radius, color, 3)
            cv2.ellipse(frame, (center_x, center_y), (radius, radius), 0, 0, 360, color, 3)
            
            # Draw checkmark
            pts = np.array([
                [center_x - 15, center_y],
                [center_x - 5, center_y + 10],
                [center_x + 20, center_y - 15]
            ], np.int32)
            cv2.polylines(frame, [pts], False, color, 3)
            
        else:
            color = (0, 0, 255)
            status_text = "❌ Face ID"
            sub_text = f"Unknown Face ({confidence:.0f}%)"
            
            # Draw X mark
            cv2.circle(frame, (center_x, center_y), radius, color, 3)
            cv2.line(frame, (center_x - 20, center_y - 20), 
                    (center_x + 20, center_y + 20), color, 3)
            cv2.line(frame, (center_x + 20, center_y - 20), 
                    (center_x - 20, center_y + 20), color, 3)
        
        # Status text
        cv2.putText(frame, status_text, (center_x - 60, center_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, sub_text, (center_x - 100, center_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Confidence bar
        bar_width = 300
        bar_height = 4
        bar_x = w//2 - bar_width//2
        bar_y = h - 50
        filled = int(bar_width * confidence)
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_height),
                     color, -1)
        
        # Controls
        cv2.putText(frame, "Press 'r' to register | 'q' to quit", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main run loop"""
        print("\n" + "="*60)
        print("  IPHONE FACE ID SYSTEM")
        print("  Style: iPhone Face ID")
        print("="*60)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ Camera not found!")
            return
        
        print("\n✅ Camera ready")
        print("\nControls:")
        print("  r - Register new user (iPhone Face ID style)")
        print("  q - Quit")
        print("\n" + "="*60)
        
        auth_success = False
        success_message_time = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Flip for mirror
                frame = cv2.flip(frame, 1)
                
                # Authenticate
                success, name, confidence = self.authenticate(frame)
                
                # Draw UI
                frame = self._draw_face_id_ui(frame, success, name, confidence)
                
                # Show success message if newly authenticated
                if success and not auth_success:
                    auth_success = True
                    success_message_time = time.time()
                    print(f"\n{'='*50}")
                    print(f"✅ SUCCESS! Your are finished!")
                    print(f"   Welcome {name}!")
                    print(f"   Confidence: {confidence:.1%}")
                    print(f"{'='*50}\n")
                    
                    # Draw big success message
                    h, w = frame.shape[:2]
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    
                    cv2.putText(frame, "SUCCESSFULLY!", (w//2 - 150, h//2 - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.putText(frame, "YOUR ARE FINISHED!", (w//2 - 180, h//2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    
                    # Show for 3 seconds
                    cv2.imshow("iPhone Face ID", frame)
                    cv2.waitKey(3000)
                    
                elif not success:
                    auth_success = False
                
                # Show success message for 3 seconds after authentication
                if auth_success and time.time() - success_message_time > 3:
                    auth_success = False
                
                cv2.imshow("iPhone Face ID", frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                    
                elif key == ord('r'):
                    name = input("\nEnter your name: ").strip()
                    if name:
                        self.register_user(name)
                        
        except KeyboardInterrupt:
            pass
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n" + "="*60)
            print("SYSTEM SHUTDOWN")
            print("="*60)


def main():
    system = IPhoneFaceID()
    system.run()


if __name__ == "__main__":
    main()