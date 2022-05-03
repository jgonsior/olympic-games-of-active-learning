from datasets.kaggle import Kaggle
from misc.config import Config

config = Config()

kaggle = Kaggle(config)
kaggle.download_all_datasets()

# TODO irgendwie hier jetzt den ten fold k cross validation split machen
