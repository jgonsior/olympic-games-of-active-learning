from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import BernoulliRBM


dense_workload_grouped = done_workload_df.groupby(
    by=[ddd for ddd in column_combinations if ddd != param_to_evaluate]
).apply(lambda r: list(zip(r["param_to_evaluate"], r["EXP_UNIQUE_ID"])))

print(f"Calculating correlations for: {len(dense_workload_grouped)}")


"""
ich iteriere darüber
und dann schaue ich zeile für zeile ob für eine Metrik (full_auc?) wie die Werte für batch_size 1,5,10 miteinander correlieren
vielleicht kann ich auch jede metrik datei der Reihe nach einlesen und kann darüber gleich das obige berechnen
"""
