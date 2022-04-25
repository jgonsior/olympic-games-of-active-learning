from typing import Dict, Tuple
import pandas as pd
from datasets.uci import UCI

# import uci_dataset as uci

from misc.config import Config

uci = UCI()


def load_gaussian_balance() -> pd.DataFrame:
    pass


def load_gaussian_unbalance() -> pd.DataFrame:
    pass


def load_xor_checkerboard() -> pd.DataFrame:
    pass


def load_d31() -> pd.DataFrame:
    pass


def load_banana() -> pd.DataFrame:
    pass


def load_phoneme() -> pd.DataFrame:
    pass


def load_ringnorme() -> pd.DataFrame:
    pass


def load_twonorm() -> pd.DataFrame:
    pass


def load_r15() -> pd.DataFrame:
    pass


def load_appendicitis() -> pd.DataFrame:
    pass


def load_ex8a() -> pd.DataFrame:
    pass


def load_ex8b() -> pd.DataFrame:
    pass


def load_dwtc() -> pd.DataFrame:
    pass


class Dataset:
    uci_datasets: Dict[int, str] = {
        0: "Sonar",
        1: "Iris",
        2: "Wine",
        3: "Parkinson",
        4: "Seeds",
        5: "Glass",
        6: "Thyroid",
        7: "Heart",
        8: "Haberman",
        9: "Ionosphere",
        10: "MUSK(Clean1)",
        11: "Breast Cancer",
        12: "Wdbc",
        13: "Statlog (Australian)",
        14: "Diabetes",
        15: "Mammograhpic",
        16: "Statlog (Vehicle)",
        17: "Tic-Tac-Toe",
        18: "Statlog (German)",
        19: "Molecular Biology (Splice)",
        20: "Phishing Websites",
        21: "Spambase",
        22: "Texture",
    }

    other_datasets: Dict[int, Tuple[str, pd.DataFrame]] = {
        23: ["Gaussian Cloud Balance", load_gaussian_balance],
        23: ["Gaussian Cloud Unbalance", load_gaussian_unbalance],
        23: ["XOR (Checkerboard2x2)", load_xor_checkerboard],
        23: ["D31", load_d31],
        23: ["Banana", load_banana],
        23: ["Phoneme", load_phoneme],
        23: ["Ringnorme", load_ringnorme],
        23: ["Twonorm", load_twonorm],
        23: ["R15", load_r15],
        23: ["Appendicitis", load_appendicitis],
        23: ["EX8b (linear)", load_ex8b],
        23: ["EX8a (non-linear)", load_ex8a],
        23: ["DWTC", load_dwtc],
    }

    def __init__(self, config: Config) -> None:
        self.config = config

    def get_dataset(self, dataset_id: int) -> pd.DataFrame:
        print(self.config.DATASETS_PATH)
        pass

    def return_dataset_statistic_table(self) -> str:
        return "not yet implemented"
