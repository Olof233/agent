from openai import AzureOpenAI
from typing import Optional

class AzureClinet():
    def __init__(self,
                model:str,
                api_version: str,
                temperature: float = 0.7,
                max_tokens: int = 1024,
                azure_endpoint: Optional[str] = None,
                api_key: Optional[str] = None,
                **kwargs):
        
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.params = kwargs
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint, #type: ignore
            api_key=self.api_key)
        


    def generate_messages(self, messages, tools=None, debug=False):
        params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **self.params
            }
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        
        

        response = self.client.chat.completions.create(**params)
        
        message = response.choices[0].message
        result = {"content": message.content or ""}
        return response, result