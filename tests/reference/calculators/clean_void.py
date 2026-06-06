import glob
import os
import numpy as np
for f in glob.glob("*.npz"):
    res = np.load(open(f, "rb"), allow_pickle=True)
    remove = False
    if "data" not in res:
        print(f"File {f} contains a void result.")
        remove = True
    elif "type" in res and res["type"] == "VoidResult":
        print(f"File {f} contains a VoidResult.")
        remove = True
    if remove:
        print(f"Removing file {f}.")
        os.remove(f)
