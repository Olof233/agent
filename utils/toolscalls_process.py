import json
import tools
import numpy as np


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def process_tool_calls(tool_calls, conversation_history):
    processed_calls = []
    tool_list = tools.get_tools()
    
    for call in tool_calls:
        name = call.get("function", {}).get("name")
        parameters = json.loads(call.get("function", {}).get("arguments", "{}"))
        
        if name in tool_list:
            result = tool_list[name].call(parameters)
            print(result)
            
            processed_calls.append({
                "name": name,
                "parameters": parameters,
                "result": result
            })
            
            conversation_history.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "name": name,
                "content": json.dumps(result, default=default_dump)
            })
        else:
            error_result = {
                "error": "ToolNotFound",
                "message": f"Tool '{name}' is not available",
                "status": "error"
            }
            
            processed_calls.append({
                "name": name,
                "parameters": parameters,
                "result": error_result
            })
            
            conversation_history.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "name": name,
                "content": json.dumps(error_result)
            })
    
    return processed_calls