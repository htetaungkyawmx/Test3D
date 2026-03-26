"""
Main entry point for Face Recognition System
"""

import os
import sys
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--camera', type=str, default='0',
                       help='Camera source (0, 1, or RTSP URL)')
    parser.add_argument('--config', type=str, default='saved_data/config.json',
                       help='Configuration file path')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU usage')
    parser.add_argument('--threshold', type=float, default=0.4,
                       help='Similarity threshold (0.3-0.7)')
    
    args = parser.parse_args()
    
    # Import and run the system
    from face_recognition_system import FaceRecognitionSystem
    
    # Update config
    config = {
        'similarity_threshold': args.threshold,
        'use_gpu': not args.no_gpu
    }
    
    # Create and run system
    system = FaceRecognitionSystem(
        camera_source=args.camera,
        config_file=args.config
    )
    
    # Update config
    for key, value in config.items():
        system.config[key] = value
    system.save_config()
    
    # Run
    system.run()

if __name__ == "__main__":
    main()