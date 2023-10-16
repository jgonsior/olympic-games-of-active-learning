from pathlib import Path
import requests

# from livereload import Server

from pandarallel import pandarallel


if __name__ == "__main__":
    from interactive_results_browser import app

    pandarallel.initialize(progress_bar=False, use_memory_fs=True)
    # check if static external ressources exist
    # if not: download them
    static_resources = {
        "https://raw.githubusercontent.com/kevquirk/simple.css/main/simple.min.css": "_simple.min.css",
        "https://cdn.jsdelivr.net/npm/tom-select@2.2.1/dist/css/tom-select.css": "_tom_min.css",
        "https://cdn.jsdelivr.net/npm/tom-select@2.2.1/dist/js/tom-select.complete.min.js": "_tom_min.js",
    }

    for sr_url, sr_local in static_resources.items():
        sr_local_path: Path = Path(f"interactive_results_browser/static/{sr_local}")
        if not sr_local_path.exists():
            sr_local_path.write_bytes(requests.get(sr_url).content)

    app.run(host="localhost")

    # server = Server(app.wsgi_app)
    # server.watch("**/*.py", ignore=True)
    # server.serve(host="0.0.0.0")
