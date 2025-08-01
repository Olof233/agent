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
    promt={'role': 'system',
            'content': '你现在是一个智能助理，用户询问的问题判断是否有可用的工具，对出现矛盾的结果将工具结果返回给用户'}
    messages={'role': 'user',
            'content': '怎么进行维护和保养. Use function retrieval.'}
    conversations.append(promt)
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