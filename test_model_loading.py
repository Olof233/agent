#!/usr/bin/env python3
"""
Test script to verify model loading works correctly
"""

from utils.vector_store import VectorStore

def test_model_loading():
    """Test if the model loads correctly"""
    print("Testing model loading...")
    
    try:
        # Test basic model loading
        vector_store = VectorStore()
        print("✓ Model loaded successfully!")
        
        # Test a simple embedding
        test_text = "Hello world"
        embedding = vector_store.model.encode([test_text])
        print(f"✓ Embedding created successfully! Shape: {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_simple_qa():
    """Test a simple QA operation"""
    print("\nTesting simple QA operation...")
    
    try:
        # Create a simple test
        test_data = {
            'question': ['What is Python?'],
            'answer': ['A programming language'],
            'score': ['9.0']
        }
        
        import pandas as pd
        df = pd.DataFrame(test_data)
        test_csv_path = 'test_simple.csv'
        df.to_csv(test_csv_path, index=False)
        
        # Test QA_RAG
        from utils.qa_rag import QA_RAG
        
        qa_rag = QA_RAG()
        index_path = qa_rag.add_qa_dataset(
            csv_path=test_csv_path,
            question_column='question',
            answer_column='answer',
            score_column='score',
            index_name='test_simple'
        )
        
        print(f"✓ QA_RAG created successfully: {index_path}")
        
        # Test search
        results = qa_rag.search_qa("programming", top_k=1)
        if results:
            print(f"✓ Search successful: {results[0]['question']}")
        
        # Cleanup
        import os
        os.remove(test_csv_path)
        
        return True
        
    except Exception as e:
        print(f"✗ QA test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Model Loading Test ===\n")
    
    success1 = test_model_loading()
    success2 = test_simple_qa()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The system is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.") 