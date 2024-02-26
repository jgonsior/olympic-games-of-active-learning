import ast
import copy
import csv
import itertools
from pathlib import Path
import sys
import glob
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing
import seaborn as sns
from sklearn.isotonic import spearmanr
from sklearn.metrics import jaccard_score

from misc.helpers import (
    append_and_create,
    get_df,
    get_glob_list,
    get_done_workload_joined_with_metric,
    save_correlation_plot,
)
from misc.plotting import set_matplotlib_size, set_seaborn_style

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()


df = get_done_workload_joined_with_metric("weighted_f1-score", config)
# df = get_done_workload_joined_with_metric("full_auc_weighted_f1-score", config)
print(df)


targets_to_evaluate = [
    "EXP_TRAIN_TEST_BUCKET_SIZE",
    "EXP_DATASET",
    "EXP_STRATEGY",
    "EXP_BATCH_SIZE",
    "EXP_LEARNER_MODEL",
    "EXP_START_POINT",
]
original_df = df.copy()
for target_to_evaluate in targets_to_evaluate:
    df = original_df.copy()
    # create fingerprints for everything EXCEPT EXP_BATCH_SIZE
    del df["EXP_UNIQUE_ID"]

    non_al_cycle_keys = [
        "EXP_DATASET",
        "EXP_STRATEGY",
        "EXP_BATCH_SIZE",
        "EXP_LEARNER_MODEL",
        "EXP_TRAIN_TEST_BUCKET_SIZE",
        "EXP_START_POINT",
    ]

    metric_keys = [kkk for kkk in df.columns if kkk not in non_al_cycle_keys]

    non_al_cycle_keys.remove(target_to_evaluate)

    # replace non_al_cycle_keys by single string fingerprint as key
    df["fingerprint"] = df[non_al_cycle_keys].apply(
        lambda row: "_".join(row.values.astype(str)), axis=1
    )

    for non_al_cycle_key in non_al_cycle_keys:
        del df[non_al_cycle_key]

    df = pd.melt(
        df, id_vars=[target_to_evaluate, "fingerprint"], value_vars=metric_keys
    )

    df["fingerprint"] = df[["fingerprint", "variable"]].apply(
        lambda row: "_".join(row.values), axis=1
    )

    del df["variable"]

    df = df.pivot(
        index="fingerprint", columns=target_to_evaluate, values="value"
    ).reset_index()

    df.columns.name = None
    df.index = df["fingerprint"]
    del df["fingerprint"]

    # TODO das hier schon eher machen (vor pivot?) um nur die Zeilen zu entfernen die na sind
    df.dropna(inplace=True)

    # print(df.corr(method="spearman"))
    # print(df.corr())

    data = df.to_numpy()
    corrmat = np.corrcoef(data.T)

    save_correlation_plot(
        data=corrmat, title=target_to_evaluate, keys=df.columns.to_list(), config=config
    )


"""
ich iteriere darüber
und dann schaue ich zeile für zeile ob für eine Metrik (full_auc?) wie die Werte für batch_size 1,5,10 miteinander correlieren
vielleicht kann ich auch jede metrik datei der Reihe nach einlesen und kann darüber gleich das obige berechnen

ja, das geht

ich will eigentlich haben 05_done_workload.csv erweitert um full_auc als neue spalte

dann lösche ich exp_unique_id
dann habe ich drei riesige zeitreihen, eine für 1, 5, 10
und als Werte jeweils die von den einzelnen fingerprints
--> und dann schaue ich halt nach, wie das ganze miteinander korreliert?

und das ganze dann auch für datasets, strategies, learner_model etc. machen



mit derselben Idee kann ich auch die Metriken (auc zumindest) evaluieeren! pro metrik eine Zeitreihe über die fingerprints
und für die standard/extended metriken klappt das ja auch, weil ich hier einfach pro fingerprint 100 werte habe

aktuell mache ich das was ich hier mache, nur schrittweise für portiönchen von immer je 100 elementen der großen zeitreihe
und dann nehme ich den Mittelwert der Zeitreihe


#####################################
grundidee:
eine zeitreihe enthält so viele werte wie ich fingerprints habe
und dann vergleiche ich die zeitreihen

und für die metriken sind die fingerprints halt die originale done_workload_df, nur dann erweitert um die ganzen metriken

Claudio fragen ob a) correlation über diese große gemittelte zeitreihe sinn macht
                    a2) nehme ich nur eine basic metric in die große gemittelte zeitreihe, oder nehme ich mehrere (was sie dann doppelt so lang machen würde)?
                  b) wie ich die basic_metrics.jpg correlation matrix interpretiere
                    ist f1_score besser als accuracy? anscheinend ja?
                  c) macht meine Idee um den Datensatz zu reduzieren so Sinn??
# each row contains fingerprint -> I want to reduce this whole thing to a correlation among these fingerprints
# I calculate for each fingerprint, how good the individual strategies are -> I save the ranking values
# in the end I get a pd.DataFrame(colums=["fingerprint", "strat_a", "strat_b", "strat_c", …])
# and in each strat_a, strat_b, strat_c column I have the single_metric result for this strategy
# then I calculate the correlation between the time series of strategy results
# claudio frage: macht es Sinn darüber zu entscheiden, welche hyperparameter combinations ich verwenden soll?

"""
