import seaborn as sns
import matplotlib.pyplot as plt


def plot_corrmatrix(df, square=True, linewidths=0.1, annot=True,
                    size=None, figsize=(12, 9), *args, **kwargs):
    """
    Plot correlation matrix of the dataset
    see doc at https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.heatmap.html#seaborn.heatmap

    """
    sns.set(context="paper", font="monospace")
    f, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.corr(), vmax=1, square=square, linewidths=linewidths,
                annot=annot, annot_kws={"size": size}, *args, **kwargs)
