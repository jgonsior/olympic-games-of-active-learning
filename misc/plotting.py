import seaborn as sns

from matplotlib import pyplot as plt

from resources.data_types import AL_STRATEGY


def set_seaborn_style(font_size=5.8, usetex=False):
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": usetex,
        # "font.family": "times",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": font_size,
        "font.size": font_size,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "xtick.bottom": True,
        "figure.titlesize": font_size,
        # "figure.autolayout": True,
    }

    plt.rcParams.update(tex_fonts)  # type: ignore

    sns.set_style("white")
    sns.set_context("paper")
    sns.set_palette("colorblind")
    plt.rcParams.update(tex_fonts)  # type: ignore


# width = 505.89
# width = 358.5049
# usage:
# fig = plt.figure(figsize=set_matplotlib_size(width, fraction=1.0))


# a
def set_matplotlib_size(width=505.89, fraction=1, half_height=False):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    if half_height:
        fig_height_in *= 0.1
    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def _rename_strategy(al_strat: str) -> str:
    al_strat = AL_STRATEGY[al_strat]

    renaming_dict = {
        AL_STRATEGY.ALIPY_RANDOM: "Random (ALIPY)",
        AL_STRATEGY.ALIPY_UNCERTAINTY_LC: "Unc. LC (ALIPY)",
        AL_STRATEGY.ALIPY_GRAPH_DENSITY: "Graph Density (ALIPY)",
        AL_STRATEGY.ALIPY_CORESET_GREEDY: "Greedy Coreset (ALIPY)",
        AL_STRATEGY.OPTIMAL_GREEDY_10: "Optimal Greedy 10",
        AL_STRATEGY.OPTIMAL_GREEDY_20: "Optimal Greedy 20",
        AL_STRATEGY.LIBACT_UNCERTAINTY_LC: "Unc. LC (LIBACT)",
        AL_STRATEGY.LIBACT_DWUS: "DWUS (LIBACT)",
        AL_STRATEGY.LIBACT_QUIRE: "QUIRE (LIBACT)",
        AL_STRATEGY.ALIPY_DENSITY_WEIGHTED: "Density Weighted (ALIPY)",
        AL_STRATEGY.ALIPY_UNCERTAINTY_MM: "Unc. MM (ALIPY)",
        AL_STRATEGY.ALIPY_UNCERTAINTY_ENTROPY: "Unc. Ent (ALIPY)",
        AL_STRATEGY.LIBACT_UNCERTAINTY_SM: "Unc. SM (LIBACT)",
        AL_STRATEGY.LIBACT_UNCERTAINTY_ENT: "Unc. Ent (LIBACT)",
        AL_STRATEGY.SMALLTEXT_LEASTCONFIDENCE: "Unc. LC (SMALLTEXT)",
        AL_STRATEGY.SMALLTEXT_PREDICTIONENTROPY: "Unc. Ent (SMALLTEXT)",
        AL_STRATEGY.SMALLTEXT_BREAKINGTIES: "Unc. BT (SMALLTEXT)",
        AL_STRATEGY.SMALLTEXT_EMBEDDINGKMEANS: "Embedding KMeans (SMALLTEXT)",
        AL_STRATEGY.SMALLTEXT_DISCRIMINATIVEAL: "Discriminative AL (SMALLTEXT)",
        AL_STRATEGY.SMALLTEXT_GREEDYCORESET: "Greedy Coreset (SMALLTEXT)",
        AL_STRATEGY.SMALLTEXT_LIGHTWEIGHTCORESET: "Lightweight Greedy Coreset (SMALLTEXT)",
        AL_STRATEGY.SMALLTEXT_CONTRASTIVEAL: "Contrastive AL (SMALLTEXT)",
        AL_STRATEGY.SMALLTEXT_RANDOM: "Random (SMALLTEXT)",
        AL_STRATEGY.SKACTIVEML_QBC: "QBC (SKACTIVEML)",
        AL_STRATEGY.SKACTIVEML_US_MARGIN: "Unc. MM (SKACTIVEML)",
        AL_STRATEGY.SKACTIVEML_US_LC: "Unc. LC (SKACTIVEML)",
        AL_STRATEGY.SKACTIVEML_US_ENTROPY: "Unc. Ent (SKACTIVEML)",
        AL_STRATEGY.SKACTIVEML_EXPECTED_AVERAGE_PRECISION: "EAP (SKACTIVEML)",
        AL_STRATEGY.SKACTIVEML_COST_EMBEDDING: "Cost Embedding (SKACTIVEML)",
        AL_STRATEGY.SKACTIVEML_DAL: "DAL (SKACTIVEML)",
        AL_STRATEGY.SKACTIVEML_MCPAL: "MCPAL (SKACTIVEML)",
        AL_STRATEGY.SKACTIVEML_QBC_VOTE_ENTROPY: "QBC (SMALLTEXT)",
        AL_STRATEGY.SKACTIVEML_QUIRE: "QUIRE (SKACTIVEML)",
    }
    print(sorted(renaming_dict.values()))
    exit(-1)

    return renaming_dict[al_strat]
