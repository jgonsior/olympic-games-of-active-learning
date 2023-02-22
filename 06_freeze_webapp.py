from pathlib import Path
import requests

from pandarallel import pandarallel
from flask_frozen import Freezer
import click
from interactive_results_browser import app

pandarallel.initialize(progress_bar=False, use_memory_fs=True)
# check if static external ressources exist
# if not: download them
static_resources = {
    "https://raw.githubusercontent.com/kevquirk/simple.css/main/simple.min.css": "_simple.min.css",
    "https://cdn.jsdelivr.net/npm/tom-select@2.2.1/dist/css/tom-select.css": "_tom_min.css",
    "https://cdn.jsdelivr.net/npm/tom-select@2.2.1/dist/js/tom-select.complete.min.js": "_tom_min.js",
}

for sr_url, sr_local_path in static_resources.items():
    sr_local_path = Path(f"interactive_results_browser/static/{sr_local_path}")
    if not sr_local_path.exists():
        sr_local_path.write_bytes(requests.get(sr_url).content)


app.config.update(FREEZER_RELATIVE_URLS=True, FREEZER_IGNORE_MIMETYPE_WARNINGS=True)
freezer = Freezer(app)

with click.progressbar(
    freezer.freeze_yield(), item_show_func=lambda p: p.url if p else "Done!"
) as urls:
    for url in urls:
        print("\n")
        print("Processing: ", url)
