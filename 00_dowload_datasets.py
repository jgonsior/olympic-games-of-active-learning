from datasets.uci import UCI


uci_datasets = UCI()

meta = uci_datasets.get_meta()
meta.to_csv("meta.csv")
