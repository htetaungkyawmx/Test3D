#!/usr/bin/env python3
"""
SIMPLE FACE LOCK SYSTEM - Working Version
Simplified face detection with clear success message
"""

import cv2
import numpy as np
import os
import pickle
import time
from datetime import datetime

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    print("❌ Please install: pip install insightface")
    exit(1)

class SimpleFaceLock:
    def __init__(self):
        # Create directories
        os.makedirs("face_data", exist_ok=True)
        
        # Load face model
        print("Loading face model...")
        self.app = FaceAnalysis(name='buffalo_l', root='saved_data')
        self.app.prepare(ctx_id=-1, det_size=(320, 320))  # Smaller for speed
        
        # Database
        self.database_file = "face_data/users.pkl"
        self.users = {}
        self.load_users()
        
        # Settings
        self.threshold = 0.75  # Lower threshold for better detection
        self.registered = False
        self.auth_success = False
        
        print("✓ System ready\n")
    
    def load_users(self):
        """Load registered users"""
        if os.path.exists(self.database_file):
            with open(self.database_file, 'rb') as f:
                self.users = pickle.load(f)
            print(f"Loaded {len(self.users)} users")
    
    def save_users(self):
        """Save users to file"""
        with open(self.database_file, 'wb') as f:
            pickle.dump(self.users, f)
    
    def get_face_embedding(self, frame):
        """Extract face embedding"""
        faces = self.app.get(frame)
        if len(faces) == 0:
            return None, None
        
        # Get best face
        best = max(faces, key=lambda x: x.det_score)
        bbox = best.bbox.astype(int)
        embedding = best.embedding
        
        return embedding, bbox
    
    def register(self, name):
        """Register new user"""
        print(f"\n📝 Registering: {name}")
        print("Look at camera...")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        embeddings = []
        captures = []
        
        # Capture 10 frames with face
        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            embedding, bbox = self.get_face_embedding(frame)
            
            if embedding is not None:
                embeddings.append(embedding)
                captures.append(frame.copy())
                
                # Draw bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"Capturing... {len(embeddings)}/30", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, "Press 'q' to cancel", (10, frame.shape[0]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Register", frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(embeddings) >= 10:
            # Average embeddings
            avg_embedding = np.mean(embeddings, axis=0)
            
            self.users[name] = {
                'embedding': avg_embedding,
                'samples': len(embeddings),
                'registered': datetime.now().isoformat()
            }
            self.save_users()
            
            print(f"\n✅ SUCCESS! User '{name}' registered!")
            print(f"   Captured {len(embeddings)} face samples")
            return True
        else:
            print(f"\n❌ Registration failed. Only captured {len(embeddings)} faces")
            return False
    
    def authenticate(self, frame):
        """Authenticate face"""
        embedding, bbox = self.get_face_embedding(frame)
        
        if embedding is None:
            return False, "No Face", 0.0, None
        
        best_match = None
        best_score = 0
        
        for name, user in self.users.items():
            stored_emb = user['embedding']
            
            # Calculate similarity
            emb_norm = embedding / np.linalg.norm(embedding)
            stored_norm = stored_emb / np.linalg.norm(stored_emb)
            score = np.dot(emb_norm, stored_norm)
            score = (score + 1) / 2  # Convert to 0-1
            
            if score > best_score:
                best_score = score
                best_match = name
        
        if best_score > self.threshold:
            return True, best_match, best_score, bbox
        else:
            return False, "Unknown", best_score, bbox
    
    def run(self):
        """Main loop"""
        print("\n" + "="*60)
        print("  SIMPLE FACE LOCK SYSTEM")
        print("="*60)
        print("\nControls:")
        print("  r - Register new user")
        print("  q - Quit")
        print("\n" + "="*60)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ Camera not found!")
            return
        
        success_shown = False
        success_time = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                
                # Authenticate
                success, name, confidence, bbox = self.authenticate(frame)
                
                # Draw result
                if success:
                    color = (0, 255, 0)
                    status = f"✅ {name}"
                    
                    # Draw bounding box
                    if bbox is not None:
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    
                    # Show success message
                    if not success_shown:
                        success_shown = True
                        success_time = time.time()
                        
                        # Draw big success message
                        h, w = frame.shape[:2]
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                        
                        cv2.putText(frame, "✅ SUCCESSFULLY!", (w//2 - 150, h//2 - 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        cv2.putText(frame, "YOUR ARE FINISHED!", (w//2 - 160, h//2 + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Welcome {name}!", (w//2 - 100, h//2 + 80),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                        print(f"\n{'='*50}")
                        print(f"✅ SUCCESS! Your are finished!")
                        print(f"   Welcome {name}!")
                        print(f"   Confidence: {confidence:.1%}")
                        print(f"{'='*50}\n")
                    
                    # Show for 3 seconds
                    if time.time() - success_time > 3:
                        success_shown = False
                        
                else:
                    color = (0, 0, 255)
                    if name == "No Face":
                        status = "❓ No Face Detected"
                    else:
                        status = f"❌ Unknown ({confidence:.0%})"
                    
                    success_shown = False
                
                # Draw status
                cv2.putText(frame, status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Draw confidence bar if not success
                if not success and name != "No Face":
                    bar_width = 200
                    bar_height = 10
                    bar_x = 10
                    bar_y = 70
                    filled = int(bar_width * confidence)
                    
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                                 (100, 100, 100), -1)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_height),
                                 color, -1)
                    cv2.putText(frame, f"Confidence: {confidence:.0%}", (bar_x, bar_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show instructions
                cv2.putText(frame, "Press 'r' to register | 'q' to quit", 
                           (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow("Face Lock System", frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    name = input("\nEnter your name: ").strip()
                    if name:
                        self.register(name)
                        
        except KeyboardInterrupt:
            pass
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nSystem shutdown")


def main():
    system = SimpleFaceLock()
    system.run()


if __name__ == "__main__":
    main()