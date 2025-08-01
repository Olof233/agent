import pandas as pd
import os
from typing import List, Dict, Any, Optional
from .vector_store import VectorStore
from .vector_search import VectorSearch


def process_csv_for_rag(csv_path: str, 
                       text_column: str,
                       index_name: Optional[str] = None,
                       additional_columns: Optional[List[str]] = None,
                       model_name: str = 'm3e-small') -> VectorSearch:
    """
    Convenience function to process a CSV file and create a RAG system
    
    Args:
        csv_path: Path to the CSV file
        text_column: Column name containing text to embed
        index_name: Custom name for the index (optional)
        additional_columns: Additional columns to store as metadata
        model_name: Sentence transformer model to use
        
    Returns:
        Configured VectorSearch instance with loaded index
    """
    # Initialize vector store system
    vector_store = VectorStore(model_name=model_name)
    
    # Add CSV data to vector store
    index_path = vector_store.upsert_csv_to_vector_store(
        csv_path=csv_path,
        text_column=text_column,
        index_name=index_name,
        additional_columns=additional_columns
    )
    
    # Initialize vector search system
    vector_search = VectorSearch(model_name=model_name)
    
    # Load the created index
    final_index_name = index_name or os.path.splitext(os.path.basename(csv_path))[0]
    if vector_search.load_index(final_index_name):
        return vector_search
    else:
        raise RuntimeError(f"Failed to load index: {index_path}")


def update_existing_rag_with_csv(vector_search: VectorSearch,
                                csv_path: str,
                                text_column: str,
                                index_name: str,
                                additional_columns: Optional[List[str]] = None) -> VectorSearch:
    """
    Update an existing RAG system with new CSV data
    
    Args:
        vector_search: Existing VectorSearch instance
        csv_path: Path to the new CSV file
        text_column: Column name containing text to embed
        index_name: Name of the index to update
        additional_columns: Additional columns to store as metadata
        
    Returns:
        Updated VectorSearch instance
    """
    # Initialize vector store for updating
    vector_store = VectorStore(model_name='m3e-small')  # Use same model
    
    # Update the vector store
    vector_store.upsert_csv_to_vector_store(
        csv_path=csv_path,
        text_column=text_column,
        index_name=index_name,
        additional_columns=additional_columns
    )
    
    # Reload the updated index
    vector_search.load_index(index_name)
    
    return vector_search


def batch_process_csv_files(csv_files: List[Dict[str, Any]], 
                           model_name: str = 'm3e-small') -> Dict[str, VectorSearch]:
    """
    Process multiple CSV files and create separate RAG systems for each
    
    Args:
        csv_files: List of dictionaries with CSV configuration
                   Each dict should have: 'path', 'text_column', 'index_name', 'additional_columns'
        model_name: Sentence transformer model to use
        
    Returns:
        Dictionary mapping index names to VectorSearch instances
    """
    rag_systems = {}
    
    for config in csv_files:
        csv_path = config['path']
        text_column = config['text_column']
        index_name = config.get('index_name')
        additional_columns = config.get('additional_columns')
        
        print(f"Processing {csv_path}...")
        
        try:
            vector_search = process_csv_for_rag(
                csv_path=csv_path,
                text_column=text_column,
                index_name=index_name,
                additional_columns=additional_columns,
                model_name=model_name
            )
            
            final_index_name = index_name or os.path.splitext(os.path.basename(csv_path))[0]
            rag_systems[final_index_name] = vector_search
            
            print(f"Successfully created RAG system for {final_index_name}")
            
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
    
    return rag_systems


def search_across_multiple_rags(rag_systems: Dict[str, VectorSearch],
                               query: str,
                               top_k_per_rag: int = 3,
                               threshold: float = 0.0) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search across multiple RAG systems and return combined results
    
    Args:
        rag_systems: Dictionary of RAG systems
        query: Search query
        top_k_per_rag: Number of results per RAG system
        threshold: Minimum similarity threshold
        
    Returns:
        Dictionary mapping RAG names to search results
    """
    results = {}
    
    for rag_name, vector_search in rag_systems.items():
        try:
            search_results = vector_search.search(query, top_k=top_k_per_rag, threshold=threshold)
            results[rag_name] = search_results
        except Exception as e:
            print(f"Error searching in {rag_name}: {e}")
            results[rag_name] = []
    
    return results


# Example usage functions
def example_job_search():
    """Example: Process job data and perform semantic search"""
    
    # Configuration for job data
    job_config = {
        'path': 'example_data/jobs/jobs_dataset.csv',
        'text_column': 'description',  # or 'positionName'
        'index_name': 'jobs_search',
        'additional_columns': ['positionName', 'company', 'location', 'salary']
    }
    
    try:
        # Process the CSV file
        vector_search = process_csv_for_rag(**job_config)
        
        # Perform searches
        queries = [
            "machine learning engineer",
            "python developer",
            "data scientist",
            "frontend developer"
        ]
        
        for query in queries:
            print(f"\nSearching for: {query}")
            print("-" * 40)
            
            results = vector_search.search(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['metadata']['positionName']} at {result['metadata']['company']}")
                print(f"   Location: {result['metadata']['location']}")
                print(f"   Similarity: {result['similarity']:.3f}")
                print()
        
        return vector_search
        
    except Exception as e:
        print(f"Error in job search example: {e}")
        return None


def example_interview_qa():
    """Example: Process interview questions and answers"""
    
    # Configuration for interview data
    interview_config = {
        'path': 'example_data/interview/deeplearning_questions.csv',
        'text_column': 'question',  # Assuming this column exists
        'index_name': 'interview_qa',
        'additional_columns': ['answer', 'category']  # Assuming these columns exist
    }
    
    try:
        # Process the CSV file
        vector_search = process_csv_for_rag(**interview_config)
        
        # Search for interview questions
        queries = [
            "neural network architecture",
            "backpropagation",
            "overfitting prevention",
            "activation functions"
        ]
        
        for query in queries:
            print(f"\nInterview Q&A for: {query}")
            print("-" * 40)
            
            results = vector_search.search(query, top_k=2)
            
            for i, result in enumerate(results, 1):
                print(f"Q{i}: {result['content']}")
                if 'answer' in result['metadata']:
                    print(f"A{i}: {result['metadata']['answer']}")
                print(f"Category: {result['metadata'].get('category', 'N/A')}")
                print(f"Similarity: {result['similarity']:.3f}")
                print()
        
        return vector_search
        
    except Exception as e:
        print(f"Error in interview Q&A example: {e}")
        return None 