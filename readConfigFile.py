import json
from types import SimpleNamespace

def readConfigFile(path):
    with open(path) as f:
        data = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    return next(iter(vars(data).values()))
