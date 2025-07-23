import copy
from .retrieval import Retrieval
from .match import Match
import json

def get_tools():
    return {
        'Retrieval': Retrieval(chunk=False),
        'Match': Match()
        }


def init_tools():
    tool_definitions = []
    tool_list = get_tools()
    for tool in tool_list.keys():
        tool_definitions.append(tool_list[tool].get_definition())
    return tool_definitions

