import seaborn as sns

def plot_corrmatrix(corr_matrix, square=True, linewidths=0.1, annot=True,
                    size=None, *args, **kwargs):
    """
    Plot correlation matrix of the dataset
    see doc at https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.heatmap.html#seaborn.heatmap

    """
    sns.heatmap(corr_matrix, vmin=corr_matrix.values.min(), vmax=1,
                square=square, linewidths=linewidths, annot=annot,
                annot_kws={"size": size}, *args, **kwargs)
