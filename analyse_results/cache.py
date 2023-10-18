from configparser import RawConfigParser


config_parser = RawConfigParser()
config_parser.read(".server_access_credentials.cfg")
cachedir = config_parser.get("LOCAL", "CACHE_DIR")

from joblib import Memory

# memory = Memory(cachedir, verbose=1)
memory = Memory(None)
