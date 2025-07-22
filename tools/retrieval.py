from sentence_transformers import SentenceTransformer
import faiss
from utils import get_pdf_text, process_file, group_lines, bert_chunk

class Retrieval():
    def __init__(self,
                name="Retrieval",
                description="Search the knowledge base",
                parameters={
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                        "required": True
                    },
                    "reference_entries": {
                        "type": "integer",
                        "description": "number of reference sentences returned",
                        "required": False
                    }},
                chunk=False    
                ):
        self.chunk = chunk
        self.model = SentenceTransformer('m3e-small')
        self.name = name
        self.description = description
        self.parameters = parameters
        self.create_kb()


    def create_kb(self):
        path = 'H3C.pdf'
        lines = process_file(get_pdf_text(path))
        content = bert_chunk(lines) if self.chunk else group_lines(lines)
        self.kb_data = content
        self.create_vector_index(content, path.split('.')[0])


    def create_vector_index(self, list, filename):
        sentence_embeddings = self.model.encode(list)
        dimension = sentence_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(sentence_embeddings) # type: ignore
        self.index_path = "index/" + filename + ".index"
        faiss.write_index(index, self.index_path)


    def search(self, query, topK=None):
        index = faiss.read_index(self.index_path)
        D, I = index.search(self.model.encode(query), topK if topK else 5)
        return I
    

    def _execute(self, parameters):
        query = parameters.get("query", "").lower()
        max_results = parameters.get("reference_entry", 5)
        if max_results == 0:
            max_results = 5
        
        if not query:
            return {"error": "Query is required"}
        
        results = []
        index = faiss.read_index(self.index_path)
        D, I = index.search(self.model.encode([query]), max_results + 5)

        for d, i in zip(D[0], I[0]):
            results.append({
                "id": i,
                "content": self.kb_data[i], # type: ignore
                "relevance": d
            })
        
        results.sort(key=lambda x: x["relevance"], reverse=True)
        results = results[:max_results]
        
        for result in results:
            del result["relevance"]
        
        return results

    
    def call(self, parameters):
        missing_params = []
        for param_name, param_def in self.parameters.items():
            if param_def.get("required", False) and param_name not in parameters:
                missing_params.append(param_name)
        
        if missing_params:
            return {
                "error": "MissingParameters",
                "message": f"Missing required parameters: {', '.join(missing_params)}",
                "status": "error"
            }
        
        result = self._execute(parameters)
        return {
            "result": result,
            "status": "success"
        }
    
    def get_definition(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": [k for k, v in self.parameters.items() 
                               if v.get("required", False)]
                }
            }
        }