from flask_caching import Cache

# hip hip hooray for python circular imports!
cache = Cache(config={"CACHE_TYPE": "FileSystemCache", "CACHE_DIR": ".cache"})
