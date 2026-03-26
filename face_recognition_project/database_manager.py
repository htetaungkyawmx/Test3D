"""
Database management for face recognition system
"""

import pickle
import json
import os
import shutil
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

class FaceDatabaseManager:
    """
    Manage face database with export/import capabilities
    """
    
    def __init__(self, database_file: str = "saved_data/face_database.pkl"):
        self.database_file = database_file
        self.backup_dir = "backups"
        
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
    
    def export_database(self, export_path: str, format: str = 'pkl'):
        """
        Export database to file
        
        Args:
            export_path: Path to export file
            format: Export format ('pkl', 'json', 'npy')
        """
        if not os.path.exists(self.database_file):
            print("No database found")
            return
        
        with open(self.database_file, 'rb') as f:
            data = pickle.load(f)
        
        if format == 'pkl':
            with open(export_path, 'wb') as f:
                pickle.dump(data, f)
        
        elif format == 'json':
            # Convert embeddings to list for JSON serialization
            json_data = {
                'names': data['names'],
                'metadata': data['metadata'],
                'embeddings': [emb.tolist() for emb in data['embeddings']]
            }
            with open(export_path, 'w') as f:
                json.dump(json_data, f, indent=4)
        
        elif format == 'npy':
            np.save(export_path, data['embeddings'])
            
        print(f"Database exported to {export_path}")
    
    def import_database(self, import_path: str, format: str = 'pkl'):
        """
        Import database from file
        
        Args:
            import_path: Path to import file
            format: Import format ('pkl', 'json', 'npy')
        """
        try:
            if format == 'pkl':
                with open(import_path, 'rb') as f:
                    data = pickle.load(f)
            
            elif format == 'json':
                with open(import_path, 'r') as f:
                    json_data = json.load(f)
                
                data = {
                    'names': json_data['names'],
                    'metadata': json_data['metadata'],
                    'embeddings': [np.array(emb) for emb in json_data['embeddings']]
                }
            
            elif format == 'npy':
                embeddings = np.load(import_path)
                data = {
                    'embeddings': embeddings,
                    'names': ['unknown'] * len(embeddings),
                    'metadata': []
                }
            
            # Backup existing database
            self.backup()
            
            # Save imported database
            with open(self.database_file, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"Database imported from {import_path}")
            
        except Exception as e:
            print(f"Import failed: {e}")
    
    def backup(self):
        """Create backup of current database"""
        if os.path.exists(self.database_file):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(self.backup_dir, f"database_backup_{timestamp}.pkl")
            shutil.copy2(self.database_file, backup_path)
            print(f"Backup created: {backup_path}")
    
    def list_faces(self) -> List[Dict]:
        """List all faces in database"""
        if not os.path.exists(self.database_file):
            return []
        
        with open(self.database_file, 'rb') as f:
            data = pickle.load(f)
        
        faces = []
        for i, name in enumerate(data['names']):
            metadata = data['metadata'][i] if i < len(data['metadata']) else {}
            faces.append({
                'index': i,
                'name': name,
                'metadata': metadata
            })
        
        return faces
    
    def remove_face(self, name: str) -> bool:
        """Remove a face from database by name"""
        if not os.path.exists(self.database_file):
            return False
        
        with open(self.database_file, 'rb') as f:
            data = pickle.load(f)
        
        # Find indices with this name
        indices = [i for i, n in enumerate(data['names']) if n == name]
        
        if not indices:
            print(f"Face '{name}' not found")
            return False
        
        # Remove in reverse order
        for idx in sorted(indices, reverse=True):
            del data['names'][idx]
            del data['embeddings'][idx]
            if idx < len(data['metadata']):
                del data['metadata'][idx]
        
        # Backup before saving
        self.backup()
        
        # Save updated database
        with open(self.database_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Removed {len(indices)} entries for '{name}'")
        return True
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the database"""
        if not os.path.exists(self.database_file):
            return {'total_faces': 0}
        
        with open(self.database_file, 'rb') as f:
            data = pickle.load(f)
        
        stats = {
            'total_faces': len(data['names']),
            'unique_names': len(set(data['names'])),
            'last_updated': data.get('last_updated', 'Unknown'),
            'embedding_dimension': len(data['embeddings'][0]) if data['embeddings'] else 0
        }
        
        return stats