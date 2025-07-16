from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',

    # required but ignored
    api_key='ollama',
)

messages=[
        {
            'role': 'user',
            'content': 'Say this is a test',
        }
    ]

chat_completion = client.chat.completions.create(
    messages=messages, # type: ignore
    model="qwen3:0.6b",
)