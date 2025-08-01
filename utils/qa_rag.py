import pandas as pd
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from .vector_store import VectorStore
from .vector_search import VectorSearch


class QA_RAG:
    """
    Specialized RAG system for Question-Answer-Score (QAS) datasets
    Stores questions as vectors and retrieves corresponding Q&A pairs
    """
    
    def __init__(self, model_name: str = 'm3e-small', index_dir: str = 'index'):
        self.vector_store = VectorStore(model_name=model_name, index_dir=index_dir)
        self.vector_search = VectorSearch(model_name=model_name, index_dir=index_dir)
        self.qa_pairs = []  # Store (Q, A, S) tuples
        self.index_name = None
    
    def add_qa_dataset(self, 
                      csv_path: str,
                      question_column: str = 'question',
                      answer_column: str = 'answer', 
                      score_column: str = 'score',
                      index_name: Optional[str] = None) -> str:
        """
        Add a QAS dataset to the vector store
        
        Args:
            csv_path: Path to the CSV file containing QAS data
            question_column: Column name for questions
            answer_column: Column name for answers
            score_column: Column name for scores
            index_name: Custom name for the index
            
        Returns:
            Path to the created index file
        """
        # Read CSV file
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        # Validate columns exist
        required_columns = [question_column, answer_column, score_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Extract QAS data
        questions = df[question_column].astype(str).tolist()
        answers = df[answer_column].astype(str).tolist()
        scores = df[score_column].astype(str).tolist()
        
        # Store QAS pairs
        self.qa_pairs = list(zip(questions, answers, scores))
        
        # Generate index name if not provided
        if index_name is None:
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            index_name = f"{base_name}_qa"
        
        self.index_name = index_name
        
        # Create vector index using questions
        index_path = self.vector_store.upsert_csv_to_vector_store(
            csv_path=csv_path,
            text_column=question_column,
            index_name=index_name,
            additional_columns=[answer_column, score_column]
        )
        
        return index_path
    
    def add_qa_list(self, 
                   qa_list: List[Tuple[str, str, str]],
                   index_name: str) -> str:
        """
        Add QAS data from a list of tuples
        
        Args:
            qa_list: List of (question, answer, score) tuples
            index_name: Name for the index
            
        Returns:
            Path to the created index file
        """
        # Convert to DataFrame
        df = pd.DataFrame(qa_list, columns=['question', 'answer', 'score'])
        
        # Store QAS pairs
        self.qa_pairs = qa_list
        self.index_name = index_name
        
        # Create temporary CSV
        temp_csv_path = f'temp_{index_name}.csv'
        df.to_csv(temp_csv_path, index=False)
        
        try:
            # Create vector index
            index_path = self.vector_store.upsert_csv_to_vector_store(
                csv_path=temp_csv_path,
                text_column='question',
                index_name=index_name,
                additional_columns=['answer', 'score']
            )
            return index_path
        finally:
            # Clean up temporary file
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
    
    def load_qa_index(self, index_name: str) -> bool:
        """
        Load an existing QAS index
        
        Args:
            index_name: Name of the index to load
            
        Returns:
            True if successfully loaded, False otherwise
        """
        if self.vector_search.load_index(index_name):
            self.index_name = index_name
            
            # Reconstruct QAS pairs from loaded data
            self.qa_pairs = []
            for i, text in enumerate(self.vector_search.kb_data):
                if i < len(self.vector_search.metadata):
                    answer = self.vector_search.metadata[i].get('answer', '')
                    score = self.vector_search.metadata[i].get('score', '')
                    self.qa_pairs.append((text, answer, score))
            
            return True
        return False
    
    def search_qa(self, 
                  query: str, 
                  top_k: int = 5, 
                  threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar questions and return QAS pairs
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of QAS results with similarity scores
        """
        if not self.vector_search.load_index(self.index_name):
            raise ValueError("No QAS index loaded")
        
        # Search using the vector search system
        results = self.vector_search.search(query, top_k=top_k, threshold=threshold)
        
        # Format results as QAS pairs
        qa_results = []
        for result in results:
            qa_result = {
                'question': result['content'],
                'answer': result['metadata'].get('answer', ''),
                'score': result['metadata'].get('score', ''),
                'similarity': result['similarity'],
                'index': result['index']
            }
            qa_results.append(qa_result)
        
        return qa_results
    
    def get_qa_by_index(self, index: int) -> Optional[Tuple[str, str, str]]:
        """
        Get QAS pair by index
        
        Args:
            index: Index of the QAS pair
            
        Returns:
            (question, answer, score) tuple or None if index out of range
        """
        if 0 <= index < len(self.qa_pairs):
            return self.qa_pairs[index]
        return None
    
    def get_all_qa_pairs(self) -> List[Tuple[str, str, str]]:
        """
        Get all QAS pairs
        
        Returns:
            List of all (question, answer, score) tuples
        """
        return self.qa_pairs.copy()
    
    def update_qa_dataset(self, 
                         csv_path: str,
                         question_column: str = 'question',
                         answer_column: str = 'answer',
                         score_column: str = 'score') -> str:
        """
        Update existing QAS dataset with new data
        
        Args:
            csv_path: Path to the new CSV file
            question_column: Column name for questions
            answer_column: Column name for answers
            score_column: Column name for scores
            
        Returns:
            Path to the updated index file
        """
        if not self.index_name:
            raise ValueError("No existing index to update")
        
        # Update using the vector store system
        index_path = self.vector_store.upsert_csv_to_vector_store(
            csv_path=csv_path,
            text_column=question_column,
            index_name=self.index_name,
            additional_columns=[answer_column, score_column]
        )
        
        # Reload to update QAS pairs
        self.load_qa_index(self.index_name)
        
        return index_path
    
    def add_qa_pair(self, question: str, answer: str, score: str) -> bool:
        """
        Add a single QAS pair to the existing index
        
        Args:
            question: Question text
            answer: Answer text
            score: Score value
            
        Returns:
            True if successfully added, False otherwise
        """
        if not self.index_name:
            raise ValueError("No existing index to update")
        
        # Create temporary CSV with single row
        temp_data = pd.DataFrame([{
            'question': question,
            'answer': answer,
            'score': score
        }])
        
        temp_csv_path = f'temp_single_{self.index_name}.csv'
        temp_data.to_csv(temp_csv_path, index=False)
        
        try:
            # Update index
            self.vector_store.upsert_csv_to_vector_store(
                csv_path=temp_csv_path,
                text_column='question',
                index_name=self.index_name,
                additional_columns=['answer', 'score']
            )
            
            # Reload to update QAS pairs
            self.load_qa_index(self.index_name)
            return True
            
        except Exception as e:
            print(f"Error adding QAS pair: {e}")
            return False
        finally:
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the QAS dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.qa_pairs:
            return {"error": "No QAS data loaded"}
        
        total_pairs = len(self.qa_pairs)
        avg_score = 0.0
        valid_scores = 0
        
        for _, _, score in self.qa_pairs:
            try:
                avg_score += float(score)
                valid_scores += 1
            except (ValueError, TypeError):
                continue
        
        if valid_scores > 0:
            avg_score /= valid_scores
        
        return {
            "total_qa_pairs": total_pairs,
            "average_score": avg_score,
            "valid_scores": valid_scores,
            "index_name": self.index_name,
            "index_loaded": self.index_name is not None
        }
    
    def export_qa_dataset(self, output_path: str) -> bool:
        """
        Export current QAS dataset to CSV
        
        Args:
            output_path: Path for the output CSV file
            
        Returns:
            True if successfully exported, False otherwise
        """
        if not self.qa_pairs:
            return False
        
        try:
            df = pd.DataFrame(self.qa_pairs, columns=['question', 'answer', 'score'])
            df.to_csv(output_path, index=False)
            return True
        except Exception as e:
            print(f"Error exporting dataset: {e}")
            return False


# Utility functions for QAS processing
def create_qa_dataset_from_csv(csv_path: str,
                              question_column: str = 'question',
                              answer_column: str = 'answer',
                              score_column: str = 'score',
                              index_name: Optional[str] = None,
                              model_name: str = 'm3e-small') -> QA_RAG:
    """
    Convenience function to create a QA_RAG system from CSV
    
    Args:
        csv_path: Path to the CSV file
        question_column: Column name for questions
        answer_column: Column name for answers
        score_column: Column name for scores
        index_name: Custom name for the index
        model_name: Sentence transformer model to use
        
    Returns:
        Configured QA_RAG instance
    """
    qa_rag = QA_RAG(model_name=model_name)
    qa_rag.add_qa_dataset(
        csv_path=csv_path,
        question_column=question_column,
        answer_column=answer_column,
        score_column=score_column,
        index_name=index_name
    )
    return qa_rag


def batch_search_qa(qa_rag: QA_RAG, 
                   queries: List[str], 
                   top_k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """
    Perform batch search on multiple queries
    
    Args:
        qa_rag: QA_RAG instance
        queries: List of search queries
        top_k: Number of results per query
        
    Returns:
        Dictionary mapping queries to search results
    """
    results = {}
    
    for query in queries:
        try:
            query_results = qa_rag.search_qa(query, top_k=top_k)
            results[query] = query_results
        except Exception as e:
            print(f"Error searching for '{query}': {e}")
            results[query] = []
    
    return results 