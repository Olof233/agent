import json
import os

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

