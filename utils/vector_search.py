import os
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import json
import numpy as np


class VectorSearch:
    """
    Handles vector search operations: loading indices and performing searches
    """
    
    def __init__(self, model_name: str = 'm3e-small', index_dir: str = 'index'):
        self.model = self._load_model_safely(model_name)
        self.index_dir = index_dir
        self.kb_data = []
        self.metadata = []
        self.index = None
        self.index_path = None
        self.current_index_name = None

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

    def load_index(self, index_name: str) -> bool:
        """
        Load an existing index for searching
        
        Args:
            index_name: Name of the index to load
            
        Returns:
            True if successfully loaded, False otherwise
        """
        index_path = os.path.join(self.index_dir, f"{index_name}.index")
        texts_path = os.path.join(self.index_dir, f"{index_name}_texts.json")
        metadata_path = os.path.join(self.index_dir, f"{index_name}_metadata.json")
        
        if not os.path.exists(index_path):
            print(f"Index file not found: {index_path}")
            return False
        
        try:
            self.index = faiss.read_index(index_path)
            self.index_path = index_path
            self.current_index_name = index_name
            
            if os.path.exists(texts_path):
                with open(texts_path, 'r', encoding='utf-8') as f:
                    self.kb_data = json.load(f)
            else:
                self.kb_data = []
            
            self.metadata = []
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            
            print(f"Successfully loaded index for searching: {index_name}")
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search the loaded index for similar texts
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results with content, metadata, and similarity scores
        """
        if self.index is None:
            raise ValueError("No index loaded. Call load_index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if similarity >= threshold and idx < len(self.kb_data):
                result = {
                    "content": self.kb_data[idx],
                    "similarity": float(similarity),
                    "index": int(idx)
                }
                
                # Add metadata if available
                if idx < len(self.metadata):
                    result["metadata"] = self.metadata[idx]
                
                results.append(result)
        
        return results

    def search_with_metadata_filter(self, query: str, metadata_filter: Dict[str, Any], top_k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search with metadata filtering
        
        Args:
            query: Search query
            metadata_filter: Dictionary of metadata key-value pairs to filter by
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of filtered search results
        """
        # Get all results first
        all_results = self.search(query, top_k=len(self.kb_data), threshold=threshold)
        
        # Filter by metadata
        filtered_results = []
        for result in all_results:
            if "metadata" in result:
                matches_filter = True
                for key, value in metadata_filter.items():
                    if key not in result["metadata"] or result["metadata"][key] != value:
                        matches_filter = False
                        break
                
                if matches_filter:
                    filtered_results.append(result)
                    if len(filtered_results) >= top_k:
                        break
        
        return filtered_results

    def batch_search(self, queries: List[str], top_k: int = 5, threshold: float = 0.0) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform batch search on multiple queries
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            threshold: Minimum similarity threshold
            
        Returns:
            Dictionary mapping queries to search results
        """
        results = {}
        
        for query in queries:
            try:
                query_results = self.search(query, top_k=top_k, threshold=threshold)
                results[query] = query_results
            except Exception as e:
                print(f"Error searching for '{query}': {e}")
                results[query] = []
        
        return results

    def get_similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate similarity score between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Encode both texts
        embeddings = self.model.encode([text1, text2])
        faiss.normalize_L2(embeddings)
        
        # Calculate cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)

    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the currently loaded index
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {"error": "No index loaded"}
        
        return {
            "index_name": self.current_index_name,
            "total_texts": len(self.kb_data),
            "total_metadata": len(self.metadata),
            "vector_dimension": self.index.d,
            "total_vectors": self.index.ntotal,
            "index_path": self.index_path
        }

    def get_text_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get text and metadata by index
        
        Args:
            index: Index of the text
            
        Returns:
            Dictionary with text and metadata, or None if index out of range
        """
        if 0 <= index < len(self.kb_data):
            result = {
                "content": self.kb_data[index],
                "index": index
            }
            
            if index < len(self.metadata):
                result["metadata"] = self.metadata[index]
            
            return result
        return None

    def get_all_texts(self) -> List[Dict[str, Any]]:
        """
        Get all texts with their metadata
        
        Returns:
            List of all texts with metadata
        """
        results = []
        for i, text in enumerate(self.kb_data):
            result = {
                "content": text,
                "index": i
            }
            
            if i < len(self.metadata):
                result["metadata"] = self.metadata[i]
            
            results.append(result)
        
        return results 