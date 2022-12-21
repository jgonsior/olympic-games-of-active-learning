import os
from typing import Any, List
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datasets import split_dataset, load_dataset, DATASET
from misc.config import Config
from pathlib import Path
from alipy.query_strategy import QueryInstanceUncertainty
import seaborn as sns

config = Config(no_cli=True)
config.DATASETS_PATH = Path("/home/jg/datasets/clean/")
config.EXP_TRAIN_TEST_BUCKET_SIZE = 3
config.EXP_BATCH_SIZE = 5
dataset_tuple = load_dataset(DATASET(34), config)


def _list_difference(long_list: List[Any], short_list: List[Any]) -> List[Any]:
    short_set = set(short_list)
    return [i for i in long_list if not i in short_set]


(
    X,
    Y,
    train_idx,
    test_idx,
    label_idx,
    unlabel_idx,
) = split_dataset(dataset_tuple, config)

model = RandomForestClassifier(n_jobs=os.cpu_count())
"""model.fit(X[train_idx], Y[train_idx])
# prediction on test set for metrics
pred = model.predict(X[test_idx, :])  # type: ignore

current_confusion_matrix = accuracy_score(
    y_true=Y[test_idx],
    y_pred=pred,
)
print(current_confusion_matrix)
exit(-1)"""
baseline = 0.837204673301601
model.fit(X[label_idx], Y[label_idx])

al_strategy = QueryInstanceUncertainty(X=X, y=Y, measure="margin")

confusion_matrices = []

for iteration in range(0, 100):
    print(iteration)
    if len(unlabel_idx) == 0:
        break

    select_ind = al_strategy.select(
        label_index=label_idx,
        unlabel_index=unlabel_idx,
        model=model,
        batch_size=config.EXP_BATCH_SIZE,
    )
    if not isinstance(select_ind, list):
        select_ind = select_ind.tolist()

    label_idx = label_idx + select_ind

    unlabel_idx = _list_difference(unlabel_idx, select_ind)

    model.fit(X=X[label_idx, :], y=Y[label_idx])  # type: ignore

    # prediction on test set for metrics
    pred = model.predict(X[test_idx, :])  # type: ignore

    current_confusion_matrix = accuracy_score(
        y_true=Y[test_idx],
        y_pred=pred,
    )

    confusion_matrices.append(current_confusion_matrix)
print(len(label_idx))
print(len(unlabel_idx))
print(len(train_idx))
print(confusion_matrices)
metric_df = pd.DataFrame(confusion_matrices, columns=["accuracy"])
metric_df["Batches"] = metric_df.index
# print(metric_df)
graph = sns.lineplot(data=metric_df, x="Batches", y="accuracy")
graph.axhline(baseline, color="red")
plt.ylim(0, 1)
# plt.show()
plt.savefig("dwtc.pdf")


confusion_matrices = [
    0.40631761142362616,
    0.5534400692340978,
    0.4707918649935093,
    0.5629597576806578,
    0.5876244050194721,
    0.6053656425789702,
    0.6767633059281696,
    0.7018606663781912,
    0.7044569450454349,
    0.7265253137170056,
    0.719169190826482,
    0.7148420597144094,
    0.7364777152747728,
    0.7477282561661618,
    0.749459108610991,
    0.7399394201644309,
    0.7659022068368672,
    0.7628732150584163,
    0.76070964950238,
    0.7563825183903072,
    0.7710947641713544,
    0.773691042838598,
    0.771960190393769,
    0.7780181739506707,
    0.7810471657291216,
    0.7767200346170489,
    0.783210731285158,
    0.7845088706187797,
    0.7901341410644742,
    0.7966248377325833,
    0.7974902639549979,
    0.784941583729987,
    0.7953266983989615,
    0.7905668541756815,
    0.7931631328429252,
    0.7970575508437906,
    0.7922977066205106,
    0.7979229770662051,
    0.8070099524015578,
    0.7953266983989615,
    0.8052790999567286,
    0.8035482475118996,
    0.812202509736045,
    0.7996538295110341,
    0.812202509736045,
    0.8147987884032886,
    0.8096062310688014,
    0.8070099524015578,
    0.8091735179575941,
    0.8139333621808741,
    0.823453050627434,
    0.8143660752920814,
    0.8173950670705322,
    0.8178277801817395,
    0.8143660752920814,
    0.8173950670705322,
    0.8160969277369104,
    0.8208567719601904,
    0.8156642146257032,
    0.8195586326265686,
    0.8221549112938122,
    0.8152315015144959,
    0.8169623539593249,
    0.818693206404154,
    0.821722198182605,
    0.8173950670705322,
    0.8152315015144959,
    0.818693206404154,
    0.8230203375162267,
    0.8260493292946777,
    0.8230203375162267,
    0.8247511899610558,
    0.8199913457377759,
    0.8243184768498486,
    0.8191259195153613,
    0.8277801817395067,
    0.8264820424058849,
    0.8256166161834704,
    0.8303764604067503,
    0.8260493292946777,
    0.828212894850714,
    0.823453050627434,
    0.823453050627434,
    0.8247511899610558,
    0.8312418866291649,
    0.834703591518823,
    0.8290783210731285,
    0.8273474686282994,
    0.8277801817395067,
    0.8256166161834704,
    0.8256166161834704,
    0.8256166161834704,
    0.8303764604067503,
    0.8351363046300303,
    0.8290783210731285,
    0.8351363046300303,
    0.8251839030722631,
    0.8351363046300303,
    0.8308091735179576,
    0.8321073128515794,
]
