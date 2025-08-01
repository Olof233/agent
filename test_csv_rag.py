#!/usr/bin/env python3
"""
Test script for CSV RAG functionality using actual CSV data
"""

import pandas as pd
import os
from utils.vector_store import VectorStore
from utils.vector_search import VectorSearch
from utils.csv_utils import process_csv_for_rag

def test_jobs_dataset():
    """Test with the actual jobs dataset"""
    print("Testing with jobs dataset...")
    
    # Use the actual jobs dataset
    csv_path = 'example_data/jobs/jobs_dataset.csv'
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        return False
    
    try:
        # Check the structure of the jobs data
        df = pd.read_csv(csv_path)
        print(f"✓ Jobs data loaded: {len(df)} rows, columns: {list(df.columns)}")
        
        # Use VectorStore to build index
        vector_store = VectorStore()
        index_path = vector_store.upsert_csv_to_vector_store(
            csv_path=csv_path,
            text_column='description',  # Assuming this column exists
            index_name='jobs_test',
            additional_columns=['positionName', 'company', 'location', 'salary']
        )
        
        print(f"✓ Jobs index built: {index_path}")
        
        # Use VectorSearch to search
        vector_search = VectorSearch()
        if vector_search.load_index('jobs_test'):
            print("✓ Jobs index loaded for searching")
            
            # Test search
            results = vector_search.search("machine learning engineer", top_k=3)
            if results:
                print(f"✓ Search successful: {len(results)} results")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['metadata'].get('positionName', 'N/A')}")
                    print(f"     Company: {result['metadata'].get('company', 'N/A')}")
                    print(f"     Location: {result['metadata'].get('location', 'N/A')}")
                    print(f"     Similarity: {result['similarity']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Jobs dataset test failed: {e}")
        return False

def test_csv_utils_with_jobs():
    """Test csv_utils with jobs data"""
    print("\nTesting CSV utils with jobs data...")
    
    csv_path = 'example_data/jobs/jobs_dataset.csv'
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        return False
    
    try:
        # Test process_csv_for_rag utility
        vector_search = process_csv_for_rag(
            csv_path=csv_path,
            text_column='positionName',  # Use position name instead of description
            index_name='jobs_utils_test',
            additional_columns=['company', 'location', 'salary']
        )
        
        print("✓ Jobs data processed with utility function")
        
        # Test search
        results = vector_search.search("software engineer", top_k=2)
        if results:
            print(f"✓ Search successful: {len(results)} results")
            for result in results:
                print(f"  - {result['content']}")
                print(f"    Company: {result['metadata'].get('company', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ CSV utils with jobs failed: {e}")
        return False

def test_advanced_features():
    """Test advanced features with actual data"""
    print("\nTesting advanced features...")
    
    # Use a smaller subset of jobs data for faster testing
    csv_path = 'example_data/jobs/jobs_dataset.csv'
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        return False
    
    try:
        # Read a sample of the data
        df = pd.read_csv(csv_path)
        sample_df = df.head(50)  # Use first 50 rows for testing
        sample_path = 'sample_jobs_test.csv'
        sample_df.to_csv(sample_path, index=False)
        
        try:
            # Build index
            vector_store = VectorStore()
            vector_store.upsert_csv_to_vector_store(
                csv_path=sample_path,
                text_column='positionName',
                index_name='advanced_test',
                additional_columns=['company', 'location']
            )
            
            # Test advanced search features
            vector_search = VectorSearch()
            vector_search.load_index('advanced_test')
            
            # Test metadata filtering
            filtered_results = vector_search.search_with_metadata_filter(
                query="engineer",
                metadata_filter={'location': 'San Francisco'},
                top_k=3
            )
            
            print(f"✓ Metadata filtering successful: {len(filtered_results)} results")
            
            # Test batch search
            batch_results = vector_search.batch_search(
                queries=["software engineer", "data scientist", "product manager"],
                top_k=2
            )
            
            print(f"✓ Batch search successful: {len(batch_results)} queries processed")
            
            # Test similarity calculation
            similarity = vector_search.get_similarity_score(
                "software engineer",
                "data engineer"
            )
            print(f"✓ Similarity calculation: {similarity:.3f}")
            
            # Test index statistics
            stats = vector_search.get_index_statistics()
            print(f"✓ Index statistics: {stats['total_texts']} texts, {stats['vector_dimension']} dimensions")
            
        finally:
            # Clean up sample file
            if os.path.exists(sample_path):
                os.remove(sample_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Advanced features failed: {e}")
        return False

def test_interview_data():
    """Test with interview questions data"""
    print("\nTesting with interview data...")
    
    csv_path = 'example_data/interview/deeplearning_questions.csv'
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        return False
    
    try:
        # Check the structure
        df = pd.read_csv(csv_path)
        print(f"✓ Interview data loaded: {len(df)} rows, columns: {list(df.columns)}")
        
        # Use VectorStore to build index
        vector_store = VectorStore()
        index_path = vector_store.upsert_csv_to_vector_store(
            csv_path=csv_path,
            text_column='question',  # Assuming this column exists
            index_name='interview_test',
            additional_columns=['answer', 'category']  # Assuming these columns exist
        )
        
        print(f"✓ Interview index built: {index_path}")
        
        # Use VectorSearch to search
        vector_search = VectorSearch()
        if vector_search.load_index('interview_test'):
            print("✓ Interview index loaded for searching")
            
            # Test search
            results = vector_search.search("neural network", top_k=2)
            if results:
                print(f"✓ Search successful: {len(results)} results")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. Q: {result['content']}")
                    if 'answer' in result['metadata']:
                        print(f"     A: {result['metadata']['answer']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Interview data test failed: {e}")
        return False

def test_index_management():
    """Test index management features"""
    print("\nTesting index management...")
    
    try:
        vector_store = VectorStore()
        
        # List all indices
        indices = vector_store.list_indices()
        print(f"✓ Found {len(indices)} existing indices")
        
        for idx_info in indices:
            print(f"  - {idx_info['index_name']}: {idx_info['texts_count']} texts")
        
        # Get info about a specific index if it exists
        if indices:
            first_index = indices[0]['index_name']
            info = vector_store.get_index_info(first_index)
            print(f"✓ Index info for '{first_index}': {info['total_vectors']} vectors")
        
        return True
        
    except Exception as e:
        print(f"✗ Index management failed: {e}")
        return False

if __name__ == "__main__":
    print("=== CSV RAG Test with Actual Data ===\n")
    
    success1 = test_jobs_dataset()
    success2 = test_csv_utils_with_jobs()
    success3 = test_advanced_features()
    success4 = test_interview_data()
    success5 = test_index_management()
    
    if success1 and success2 and success3 and success4 and success5:
        print("\n🎉 All tests passed! CSV RAG is working correctly with actual data.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.") 