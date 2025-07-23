from sentence_transformers import SentenceTransformer
import faiss
from utils import get_pdf_text, process_file, group_lines, bert_chunk, read_json
import os

SPECIAL_LIST = ['positionName', 'description']

class Match():
    def __init__(self,
                name="Match",
                description="Match the key word",
                parameters={
                    "key word": {
                        "type": "string",
                        "description": "Query string of categories",
                        "required": True
                    },
                    "categories": {
                        "type": "string",
                        "enum": ['company', 'rating', 'location',\
                                'positionName', 'description', 'salary', 'jobType'],
                        "description": "Categories of key word, 'full-time' and 'Contract' are the only two values of jobType, rating must be numerical",
                        "required": True
                    }
                }
    ):
        self.model = SentenceTransformer('m3e-small')
        self.name = name
        self.description = description
        self.parameters = parameters
        self.create_jsonkb()



    def create_jsonkb(self):
        path = 'example_data/jobs/data.json'
        self.kb_data = read_json(path)
        filename = path.split('/')[-1].split('.')[0]
        self.positionName_path = "index/" + filename + "_positionName.index"
        self.description_path = "index/" + filename + "_description.index"
        if not os.path.isfile(self.positionName_path):
            positionNames = []
            for item in self.kb_data:
                positionNames.append(item['positionName'])
            self.create_vector_index(positionNames, self.positionName_path)
        if not os.path.isfile(self.description_path):
            descriptions = []
            for item in self.kb_data:
                descriptions.append(item['description'])
            self.create_vector_index(descriptions, self.description_path)


    def create_vector_index(self, list, path):
        sentence_embeddings = self.model.encode(list)
        dimension = sentence_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(sentence_embeddings) # type: ignore
        faiss.write_index(index, path)
    

    def _execute(self, parameters):
        query = parameters.get("key word", "").lower()
        max_results = parameters.get("reference_entry", 3)
        if max_results == 0:
            max_results = 3
        
        if not query:
            return {"error": "key word is required"}
        
        results = []

        attr = parameters.get('categories')
        if attr in SPECIAL_LIST:
            path = self.description_path if attr.lower() == 'description' else self.positionName_path
            index = faiss.read_index(path)
            D, I = index.search(self.model.encode([query]), max_results + 2)

            for d, i in zip(D[0], I[0]):
                results.append({
                    "content": self.kb_data[i],
                    "relevance": d
                })
            
            results.sort(key=lambda x: x["relevance"], reverse=True)
            results = results[:max_results]
            
            for result in results:
                del result["relevance"]

        else:
            for item in self.kb_data:
                for words in item[attr].lower().split(' '):
                    if words in parameters.get('key word').lower():
                        results.append(item)
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
        