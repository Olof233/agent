# Migration Guide: From Faiss_RAG to Separated Architecture

## 🚀 **Overview**

We've refactored the RAG system to separate concerns into two distinct classes:
- **`VectorStore`**: Handles building, updating, and managing indices
- **`VectorSearch`**: Handles loading indices and performing searches

## 📋 **Migration Summary**

| Old Code | New Code |
|----------|----------|
| `from utils.rag import Faiss_RAG` | `from utils.vector_store import VectorStore`<br>`from utils.vector_search import VectorSearch` |
| `rag = Faiss_RAG()` | `vector_store = VectorStore()`<br>`vector_search = VectorSearch()` |
| `rag.upsert_csv_to_vector_store()` | `vector_store.upsert_csv_to_vector_store()` |
| `rag.load_index()` | `vector_search.load_index()` |
| `rag.search()` | `vector_search.search()` |

## 🔄 **Step-by-Step Migration**

### **1. Building/Updating Indices**

**Before:**
```python
from utils.rag import Faiss_RAG

rag = Faiss_RAG()
index_path = rag.upsert_csv_to_vector_store(
    csv_path='data.csv',
    text_column='text',
    index_name='my_index'
)
```

**After:**
```python
from utils.vector_store import VectorStore

vector_store = VectorStore()
index_path = vector_store.upsert_csv_to_vector_store(
    csv_path='data.csv',
    text_column='text',
    index_name='my_index'
)
```

### **2. Searching Indices**

**Before:**
```python
rag.load_index('my_index')
results = rag.search("query", top_k=5)
```

**After:**
```python
from utils.vector_search import VectorSearch

vector_search = VectorSearch()
vector_search.load_index('my_index')
results = vector_search.search("query", top_k=5)
```

### **3. Advanced Search Features**

**Before:** Not available

**After:**
```python
# Metadata filtering
filtered_results = vector_search.search_with_metadata_filter(
    query="learning",
    metadata_filter={'category': 'AI'},
    top_k=3
)

# Batch search
batch_results = vector_search.batch_search(
    queries=["query1", "query2", "query3"],
    top_k=2
)

# Similarity calculation
similarity = vector_search.get_similarity_score("text1", "text2")
```

## 🎯 **QA_RAG Usage (No Changes Needed)**

The `QA_RAG` class has been updated internally to use the new architecture, so existing code continues to work:

```python
from utils.qa_rag import QA_RAG

# This still works exactly the same
qa_rag = QA_RAG()
qa_rag.add_qa_dataset(csv_path='qa_data.csv')
results = qa_rag.search_qa("question")
```

## 🛠️ **Utility Functions**

The `csv_utils.py` functions have been updated to return `VectorSearch` instances instead of `Faiss_RAG`:

```python
from utils.csv_utils import process_csv_for_rag

# Returns VectorSearch instance instead of Faiss_RAG
vector_search = process_csv_for_rag(
    csv_path='data.csv',
    text_column='text'
)

# Use for searching
results = vector_search.search("query")
```

## 📊 **Benefits of the New Architecture**

### **1. Separation of Concerns**
- **VectorStore**: Only handles building/updating operations
- **VectorSearch**: Only handles search operations
- **QA_RAG**: Coordinates between them for QAS-specific tasks

### **2. Better Flexibility**
```python
# Build indices without loading them for search
vector_store = VectorStore()
vector_store.upsert_csv_to_vector_store(csv_path='data.csv')

# Search existing indices without building capabilities
vector_search = VectorSearch()
vector_search.load_index('existing_index')
results = vector_search.search("query")
```

### **3. Enhanced Features**
- Metadata filtering
- Batch searching
- Similarity calculations
- Index statistics
- Better error handling

### **4. Improved Testing**
- Test storage operations separately
- Test search operations separately
- Easier to mock individual components

## 🔧 **Backward Compatibility**

- ✅ `QA_RAG` maintains the same interface
- ✅ `csv_utils` functions work the same way
- ✅ All existing functionality is preserved
- ✅ New features are additive

## 🚨 **Breaking Changes**

- ❌ `Faiss_RAG` class has been removed
- ❌ Direct imports from `utils.rag` will fail
- ❌ `csv_utils` functions now return `VectorSearch` instead of `Faiss_RAG`

## 📝 **Quick Migration Checklist**

- [ ] Update imports from `utils.rag` to `utils.vector_store` and `utils.vector_search`
- [ ] Replace `Faiss_RAG()` with `VectorStore()` for building operations
- [ ] Replace `Faiss_RAG()` with `VectorSearch()` for search operations
- [ ] Update any code that expects `Faiss_RAG` instances from utility functions
- [ ] Test your existing functionality
- [ ] Consider using new advanced features like metadata filtering

## 🆘 **Need Help?**

If you encounter any issues during migration:

1. Check that all imports are updated
2. Verify that you're using the correct class for your use case
3. Test with the provided test files: `test_model_loading.py`, `test_csv_rag.py`
4. Review the example files for usage patterns

The new architecture provides the same functionality with better organization and additional features! 🎉 