from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification

X, Y = make_classification(
    n_samples=1000,
    n_features=2,
    n_classes=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
)

df = pd.DataFrame(X)
df["Y"] = Y
df.plot.scatter(x=0, y=1, c="Y", colormap="viridis")
plt.savefig("balance.jpg")
plt.clf()
df.to_csv("gaussian_balanced.csv")

X, Y = make_classification(
    n_samples=1000,
    n_features=2,
    n_classes=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    weights=[1, 2],
)

df = pd.DataFrame(X)
df["Y"] = Y
df.plot.scatter(x=0, y=1, c="Y", colormap="viridis")
plt.savefig("unbalance.jpg")

df.to_csv("gaussian_unbalanced.csv")
print(df)
