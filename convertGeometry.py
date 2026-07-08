import cupy as cp
from types import SimpleNamespace

def convertGeometry(tasElements):

    idxList = range(len(tasElements))  # 0-based indices

    tasElementsRestructured = [SimpleNamespace() for _ in idxList]

    # add transducer infos for emitter
    for i in idxList:
        tasElementsRestructured[i].emitterPositions = tasElements[i].transducerPositions
        tasElementsRestructured[i].emitterNormals = tasElements[i].transducerNormals

    # add transducer infos for receiver
    for i in idxList:
        tasElementsRestructured[i].receiverPositions = tasElements[i].transducerPositions
        tasElementsRestructured[i].receiverNormals = tasElements[i].transducerNormals

    return tasElementsRestructured
