import json
from types import SimpleNamespace

def readConfigFile(path):
    with open(path) as f:
        return json.load(f, object_hook=lambda d: SimpleNamespace(**d))