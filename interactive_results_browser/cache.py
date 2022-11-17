cachedir = ".cache"

from joblib import Memory

memory = Memory(cachedir, verbose=1)
