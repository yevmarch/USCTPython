from writeReconstructionLog import writeReconstructionLog

def compareUniqueIDs(rootUniqueID, currentUniqueID):

    if not isinstance(rootUniqueID, str) or not rootUniqueID:
        raise ValueError("'path' must be a non-empty string")
    if not isinstance(currentUniqueID, str):
        raise ValueError("'name' must be a string")
    
    # strip whitespaces
    currentUniqueID = currentUniqueID.strip()
    rootUniqueID = rootUniqueID.strip()
    
    # compare IDs
    if rootUniqueID != currentUniqueID:
        msg = 'UniqueIDs are not equal. Stopping reconstructions'
        writeReconstructionLog(msg, 4)
        raise ValueError(f"compareUniqueIDs:uIDsNotMatching - {msg}")