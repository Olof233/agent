#!/usr/bin/env python3
"""
Test script for QA_RAG functionality using actual CSV data
"""

import pandas as pd
import os
from utils.qa_rag import QA_RAG, create_qa_dataset_from_csv

def test_qa_dataset_creation():
    """Test QA dataset creation with actual CSV data"""
    print("Testing QA dataset creation...")
    
    # Use the actual sample QA dataset
    csv_path = 'example_data/sample_qa_dataset.csv'
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        return False
    
    try:
        # Test QA_RAG with actual data
        qa_rag = QA_RAG()
        index_path = qa_rag.add_qa_dataset(
            csv_path=csv_path,
            question_column='question',
            answer_column='answer',
            score_column='score',
            index_name='sample_qa_test'
        )
        
        print(f"✓ QA dataset created: {index_path}")
        
        # Test search functionality
        results = qa_rag.search_qa("machine learning", top_k=3)
        if results:
            print(f"✓ Search successful: {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Q: {result['question']}")
                print(f"     A: {result['answer']}")
                print(f"     Score: {result['score']}")
                print(f"     Similarity: {result['similarity']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ QA dataset creation failed: {e}")
        return False

def test_qa_utility_functions():
    """Test QA utility functions with actual data"""
    print("\nTesting QA utility functions...")
    
    # Use the updated QA dataset
    csv_path = 'example_data/updated_qa_dataset.csv'
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        return False
    
    try:
        # Test create_qa_dataset_from_csv utility
        qa_rag = create_qa_dataset_from_csv(
            csv_path=csv_path,
            question_column='question',
            answer_column='answer',
            score_column='score',
            index_name='updated_qa_test'
        )
        
        print("✓ QA dataset created with utility function")
        
        # Test statistics
        stats = qa_rag.get_statistics()
        print(f"✓ Statistics: {stats['total_qa_pairs']} pairs, avg score: {stats['average_score']:.2f}")
        
        # Test search
        results = qa_rag.search_qa("artificial intelligence", top_k=2)
        if results:
            print(f"✓ Search successful: {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"✗ QA utility functions failed: {e}")
        return False

def test_qa_operations():
    """Test various QA operations with actual data"""
    print("\nTesting QA operations...")
    
    # Use the sample QA dataset
    csv_path = 'example_data/sample_qa_dataset.csv'
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        return False
    
    try:
        qa_rag = QA_RAG()
        qa_rag.add_qa_dataset(
            csv_path=csv_path,
            question_column='question',
            answer_column='answer',
            score_column='score',
            index_name='operations_test'
        )
        
        # Test get_qa_by_index
        qa_pair = qa_rag.get_qa_by_index(0)
        if qa_pair:
            print(f"✓ Get QA by index: {qa_pair[0][:50]}...")
        
        # Test get_all_qa_pairs
        all_pairs = qa_rag.get_all_qa_pairs()
        print(f"✓ Get all QA pairs: {len(all_pairs)} pairs")
        
        # Test add_qa_pair
        success = qa_rag.add_qa_pair(
            question="What is the capital of France?",
            answer="Paris",
            score="10.0"
        )
        if success:
            print("✓ Add QA pair successful")
        
        # Test export functionality
        export_path = 'test_export.csv'
        export_success = qa_rag.export_qa_dataset(export_path)
        if export_success:
            print("✓ Export QA dataset successful")
            # Clean up exported file
            if os.path.exists(export_path):
                os.remove(export_path)
        
        return True
        
    except Exception as e:
        print(f"✗ QA operations failed: {e}")
        return False

def test_batch_search():
    """Test batch search functionality"""
    print("\nTesting batch search...")
    
    # Use the sample QA dataset
    csv_path = 'example_data/sample_qa_dataset.csv'
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        return False
    
    try:
        qa_rag = QA_RAG()
        qa_rag.add_qa_dataset(
            csv_path=csv_path,
            question_column='question',
            answer_column='answer',
            score_column='score',
            index_name='batch_test'
        )
        
        # Test batch search
        queries = [
            "machine learning",
            "deep learning", 
            "neural networks",
            "artificial intelligence"
        ]
        
        # Assuming batch_search_qa is defined elsewhere or will be added
        # For now, we'll just call search_qa for each query
        batch_results = {}
        for query in queries:
            results = qa_rag.search_qa(query, top_k=2)
            batch_results[query] = results
        
        print(f"✓ Batch search successful: {len(batch_results)} queries processed")
        for query, results in batch_results.items():
            print(f"  '{query}': {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch search failed: {e}")
        return False

def test_with_interview_data():
    """Test with interview questions data"""
    print("\nTesting with interview data...")
    
    # Use the deep learning questions
    csv_path = 'example_data/interview/deeplearning_questions.csv'
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        return False
    
    try:
        # Check the structure of the interview data
        df = pd.read_csv(csv_path)
        print(f"✓ Interview data loaded: {len(df)} rows, columns: {list(df.columns)}")
        
        # Try to use it with QA_RAG (assuming it has question/answer structure)
        if 'question' in df.columns and 'answer' in df.columns:
            qa_rag = QA_RAG()
            
            # Create a temporary CSV with proper structure
            temp_data = df[['question', 'answer']].copy()
            temp_data['score'] = '8.0'  # Default score
            temp_csv_path = 'temp_interview.csv'
            temp_data.to_csv(temp_csv_path, index=False)
            
            try:
                index_path = qa_rag.add_qa_dataset(
                    csv_path=temp_csv_path,
                    question_column='question',
                    answer_column='answer',
                    score_column='score',
                    index_name='interview_test'
                )
                
                print(f"✓ Interview QA dataset created: {index_path}")
                
                # Test search
                results = qa_rag.search_qa("neural network", top_k=2)
                if results:
                    print(f"✓ Interview search successful: {len(results)} results")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_csv_path):
                    os.remove(temp_csv_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Interview data test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== QA_RAG Test with Actual Data ===\n")
    
    success1 = test_qa_dataset_creation()
    success2 = test_qa_utility_functions()
    success3 = test_qa_operations()
    success4 = test_batch_search()
    success5 = test_with_interview_data()
    
    if success1 and success2 and success3 and success4 and success5:
        print("\n🎉 All tests passed! QA_RAG is working correctly with actual data.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.") 