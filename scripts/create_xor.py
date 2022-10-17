from pathlib import Path
import urllib.request
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

urls_to_download = [
    "checkerboard2x2_",
    "checkerboard4x4_",
    "rotated_checkerboard2x2_",
]

for url in urls_to_download:
    print(f"Downloading {url}")
    url_train = (
        "https://github.com/ksenia-konyushkova/LAL/raw/master/data/" + url + "train.npz"
    )
    url_test = (
        "https://github.com/ksenia-konyushkova/LAL/raw/master/data/" + url + "test.npz"
    )
    urllib.request.urlretrieve(url_train, "train.npz")
    urllib.request.urlretrieve(url_test, "test.npz")

    train_data = np.load("train.npz")
    test_data = np.load("test.npz")

    train_X = train_data["x"]
    train_Y = train_data["y"]
    test_X = test_data["x"]
    test_Y = test_data["y"]

    """
    df = pd.DataFrame(train_X)
    df["Y"] = train_Y
    print(df)
    df.plot.scatter(x=0, y=1, c="Y", colormap="viridis")
    plt.savefig(url + "_train.jpg")
    plt.clf()

    df = pd.DataFrame(test_X)
    df["Y"] = test_Y
    print(df)
    df.plot.scatter(x=0, y=1, c="Y", colormap="viridis")
    plt.savefig(url + "_test.jpg")
    plt.clf()
    """
    df = pd.DataFrame(np.concatenate((test_X, train_X)))
    df["Y"] = np.concatenate((test_Y, train_Y))

    """print(df)
    df.plot.scatter(x=0, y=1, c="Y", colormap="viridis")
    plt.savefig(url + "_both.jpg")
    plt.clf()"""

    df.to_csv(url + ".csv", index=None)

    # delete file
    t = Path("test.npz")
    t.unlink()
    t = Path("train.npz")
    t.unlink()
