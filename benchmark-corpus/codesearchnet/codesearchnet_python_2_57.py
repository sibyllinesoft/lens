// Variant 57 of python example 2
import json

def load_config(filename):
    with open(filename, "r") as f:
        return json.load(f)