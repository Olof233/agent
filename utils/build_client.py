import yaml
from models import AzureClinet
from models import OllamaClient
import os

def read_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    

def get_model_client(provider: str, model_name: str, **kwargs):
    if provider.lower() == "azure":
        kwargs.pop('base_url', None)
        return AzureClinet(model=model_name, **kwargs)
    elif provider.lower() == "ollama":
        kwargs.pop('endpoint', None)
        kwargs.pop('api_version', None)
        return OllamaClient(model=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported model provider: {provider}")


def load_models_from_config(config):
    models = []
    for model_config in config.get('models', []):
        provider = model_config.get('provider')
        model_name = model_config.get('name')
        
        api_key = os.environ.get(f"{provider.upper()}_API_KEY", model_config.get('api_key'))
        if not api_key:
            api_key = True
        
        if api_key:
            model = get_model_client(
                provider=provider,
                model_name=model_name,
                api_key=api_key,
                temperature=model_config.get('temperature', 0.7),
                max_tokens=model_config.get('max_tokens', 1024),
                endpoint=model_config.get('azure_endpoint'),
                api_version=model_config.get('api_version'),
                base_url=model_config.get('base_url'),
                **(model_config.get('parameters', {}))
            )
            models.append(model)
    
    return models