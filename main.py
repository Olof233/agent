import utils
import models
import tools

path = "config/models.yaml"

def main(path):

    #initialize settings
    config = utils.read_yaml(path)
    models_list = utils.load_models_from_config(config)
    print(f"Loaded models: {models_list}") 
    tool_definitions = tools.init_tools()

    #initialize conversation
    conversations = []
    messages={'role': 'user',
            'content': '在你的知识库中查询，KylinOS 操作系统安装nvidia-container-runtime /no_think'}
    conversations.append(messages)

    #generate response 
    for model in models_list:
        response, result = model.generate_messages(conversations, tools=tool_definitions)
        new_content = {
            "role": "assistant",
            "content": result.get("content", ""),
            **({} if "tool_calls" not in result else {"tool_calls": result["tool_calls"]})}
        conversations.append(new_content)        
        print(response)
        if "tool_calls" in result:
            print(utils.process_tool_calls(result["tool_calls"], conversations))
            response, result = model.generate_messages(messages=conversations)
            print(f"Final Response:", response)



if __name__ == "__main__":
    main(path)