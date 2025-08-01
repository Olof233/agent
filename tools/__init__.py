import copy
from .pdf_retrieval import Retrieval
from .jobs_match import Match
import json
import copy


def get_tools():
    return {
        'Retrieval': Retrieval(chunk=False),
        'Match': Match()
        }


def remove_required_from_properties(tool_definitions):
    new_tool_definitions = copy.deepcopy(tool_definitions)
    for tool in new_tool_definitions:
        properties = tool.get('function', {}).get('parameters', {}).get('properties', {})
        for prop_name, prop in properties.items():
            if 'required' in prop:
                del prop['required']
    return new_tool_definitions


def init_tools():
    tool_definitions = []
    tool_list = get_tools()
    for tool in tool_list.keys():
        tool_definitions.append(tool_list[tool].get_definition())
    return remove_required_from_properties(tool_definitions)

