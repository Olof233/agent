import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import json
import torch


class VectorStore:
    """
    Handles vector storage operations: building, updating, and managing indices
    """
    
    def __init__(self, model_name: str = 'm3e-small', index_dir: str = 'index'):
        self.model = self._load_model_safely(model_name)
        self.index_dir = index_dir
        self.kb_data = []
        self.metadata = []
        self.index_path = None
        os.makedirs(index_dir, exist_ok=True)

    def _load_model_safely(self, model_name: str) -> SentenceTransformer:
        """
        Safely load the sentence transformer model with fallback options
        """
        try:
            # Try loading with default settings first
            return SentenceTransformer(model_name)
        except Exception as e:
            print(f"Warning: Failed to load {model_name} with default settings: {e}")
            
            # Try alternative models if m3e-small fails
            fallback_models = [
                'all-MiniLM-L6-v2',  # Smaller, more compatible model
                'paraphrase-MiniLM-L3-v2',  # Another lightweight option
                'distiluse-base-multilingual-cased-v2'  # Multilingual fallback
            ]
            
            for fallback_model in fallback_models:
                try:
                    print(f"Trying fallback model: {fallback_model}")
                    model = SentenceTransformer(fallback_model)
                    print(f"Successfully loaded fallback model: {fallback_model}")
                    return model
                except Exception as fallback_error:
                    print(f"Fallback model {fallback_model} also failed: {fallback_error}")
                    continue
            
            # If all else fails, try to load with specific settings
            try:
                print("Attempting to load with custom settings...")
                # Try with specific device and trust_remote_code settings
                model = SentenceTransformer(
                    model_name,
                    device='cpu',  # Force CPU to avoid GPU issues
                    trust_remote_code=True
                )
                return model
            except Exception as final_error:
                raise RuntimeError(f"Failed to load any sentence transformer model. "
                                 f"Original error: {e}. "
                                 f"Final attempt error: {final_error}. "
                                 f"Please check your internet connection and try again.")

    def _extract_texts_and_metadata_from_csv(self, csv_path: str, text_column: str, additional_columns: Optional[List[str]] = None):
        """Extract texts and metadata from CSV file"""
        df = pd.read_csv(csv_path, encoding='utf-8')
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV file")
        texts = df[text_column].astype(str).tolist()
        metadata = []
        if additional_columns:
            for _, row in df.iterrows():
                meta = {col: str(row[col]) for col in additional_columns if col in df.columns}
                metadata.append(meta)
        return texts, metadata

    def upsert_csv_to_vector_store(self, csv_path: str, text_column: str, index_name: Optional[str] = None, additional_columns: Optional[List[str]] = None) -> str:
        """
        Add or update CSV file data to the vector store. If the index exists, append new data; otherwise, create a new index.
        """
        # Load existing data if index exists
        if index_name is None:
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            index_name = f"{base_name}_{text_column}"
        
        if self.load_index(index_name):
            existing_texts = self.kb_data
            existing_metadata = self.metadata
        else:
            existing_texts, existing_metadata = [], []
        
        # Extract new data
        new_texts, new_metadata = self._extract_texts_and_metadata_from_csv(csv_path, text_column, additional_columns)
        
        # Combine and update
        combined_texts = existing_texts + new_texts
        combined_metadata = existing_metadata + (new_metadata or [{}] * len(new_texts))
        
        index_path = self._create_vector_index(combined_texts, index_name, combined_metadata)
        self.load_index(index_name)
        return index_path

    def _create_vector_index(self, texts: List[str], index_name: str, metadata: Optional[List[Dict[str, Any]]] = None) -> str:
        """Create a FAISS vector index from text data"""
        # Filter out empty texts
        valid_texts = []
        valid_metadata = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text.strip())
                if metadata and i < len(metadata):
                    valid_metadata.append(metadata[i])
                else:
                    valid_metadata.append({})
        
        if not valid_texts:
            raise ValueError("No valid texts found for embedding")
        
        # Create embeddings
        print(f"Creating embeddings for {len(valid_texts)} texts...")
        embeddings = self.model.encode(valid_texts, show_progress_bar=True)
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Save index
        index_path = os.path.join(self.index_dir, f"{index_name}.index")
        faiss.write_index(index, index_path)
        
        # Save metadata and texts
        if valid_metadata:
            metadata_path = os.path.join(self.index_dir, f"{index_name}_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(valid_metadata, f, ensure_ascii=False, indent=2)
        
        texts_path = os.path.join(self.index_dir, f"{index_name}_texts.json")
        with open(texts_path, 'w', encoding='utf-8') as f:
            json.dump(valid_texts, f, ensure_ascii=False, indent=2)
        
        print(f"Vector index saved to: {index_path}")
        return index_path
    
    def load_index(self, index_name: str) -> bool:
        """Load an existing index into memory"""
        index_path = os.path.join(self.index_dir, f"{index_name}.index")
        texts_path = os.path.join(self.index_dir, f"{index_name}_texts.json")
        metadata_path = os.path.join(self.index_dir, f"{index_name}_metadata.json")
        
        if not os.path.exists(index_path):
            print(f"Index file not found: {index_path}")
            return False
        
        try:
            self.index = faiss.read_index(index_path)
            self.index_path = index_path
            
            if os.path.exists(texts_path):
                with open(texts_path, 'r', encoding='utf-8') as f:
                    self.kb_data = json.load(f)
            else:
                self.kb_data = []
            
            self.metadata = []
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            
            print(f"Successfully loaded index: {index_name}")
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def get_index_info(self, index_name: str) -> Dict[str, Any]:
        """Get information about an index"""
        index_path = os.path.join(self.index_dir, f"{index_name}.index")
        texts_path = os.path.join(self.index_dir, f"{index_name}_texts.json")
        metadata_path = os.path.join(self.index_dir, f"{index_name}_metadata.json")
        
        info = {
            "index_name": index_name,
            "index_path": index_path,
            "exists": os.path.exists(index_path),
            "texts_count": 0,
            "has_metadata": False
        }
        
        if os.path.exists(index_path):
            try:
                index = faiss.read_index(index_path)
                info["vector_dimension"] = index.d
                info["total_vectors"] = index.ntotal
            except Exception as e:
                info["error"] = str(e)
        
        if os.path.exists(texts_path):
            try:
                with open(texts_path, 'r', encoding='utf-8') as f:
                    texts = json.load(f)
                info["texts_count"] = len(texts)
            except Exception as e:
                info["texts_error"] = str(e)
        
        if os.path.exists(metadata_path):
            info["has_metadata"] = True
        
        return info
    
    def list_indices(self) -> List[Dict[str, Any]]:
        """List all available indices"""
        indices = []
        
        if not os.path.exists(self.index_dir):
            return indices
        
        for filename in os.listdir(self.index_dir):
            if filename.endswith('.index'):
                index_name = filename[:-6]  # Remove '.index' extension
                indices.append(self.get_index_info(index_name))
        
        return indices
    
    def delete_index(self, index_name: str) -> bool:
        """Delete an index and its associated files"""
        try:
            index_path = os.path.join(self.index_dir, f"{index_name}.index")
            texts_path = os.path.join(self.index_dir, f"{index_name}_texts.json")
            metadata_path = os.path.join(self.index_dir, f"{index_name}_metadata.json")
            
            files_to_delete = [index_path, texts_path, metadata_path]
            
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            print(f"Successfully deleted index: {index_name}")
            return True
            
        except Exception as e:
            print(f"Error deleting index {index_name}: {e}")
            return False 