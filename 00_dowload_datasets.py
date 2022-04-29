from datasets.kaggle import Kaggle
import pandas as pd
from config.config import Config
from misc.logging import log_it

config = Config()

kaggle = Kaggle(config)
kaggle.download_all_datasets()
