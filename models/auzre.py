from openai import AzureOpenAI
from typing import Optional

class AzureClinet():
    def __init__(self,
                model:str,
                temperature: float = 0.7,
                max_tokens: int = 1024,
                api_version: Optional[str] = None,
                endpoint: Optional[str] = None,
                api_key: Optional[str] = None,
                **kwargs):
        
        self.api_version = api_version
        self.azure_endpoint = endpoint
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

        if hasattr(message, "tool_calls") and message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in message.tool_calls
            ]

        return response, result