import json
import os

def read_json(path):
    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

