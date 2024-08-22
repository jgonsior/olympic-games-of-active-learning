import seaborn as sns

from matplotlib import pyplot as plt

from resources.data_types import AL_STRATEGY


def set_seaborn_style(font_size=5.8, usetex=False):
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": usetex,
        "font.family": "times",
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
        AL_STRATEGY.ALIPY_RANDOM: "Rand (ALI)",
        AL_STRATEGY.ALIPY_UNCERTAINTY_LC: "LC (ALI)",
        AL_STRATEGY.ALIPY_GRAPH_DENSITY: "GD (ALI)",
        AL_STRATEGY.ALIPY_CORESET_GREEDY: "Core (ALI)",htop
        AL_STRATEGY.OPTIMAL_GREEDY_10: "OG10",
        AL_STRATEGY.OPTIMAL_GREEDY_20: "OG20",
        AL_STRATEGY.LIBACT_UNCERTAINTY_LC: "LC (LIB)",
        AL_STRATEGY.LIBACT_DWUS: "DWUS (LIB)",
        AL_STRATEGY.LIBACT_QUIRE: "QUIRE (LIB)",
        AL_STRATEGY.ALIPY_DENSITY_WEIGHTED: "DWUS (ALI)",
        AL_STRATEGY.ALIPY_UNCERTAINTY_MM: "MM (ALI)",
        AL_STRATEGY.ALIPY_UNCERTAINTY_ENTROPY: " Ent (ALI)",
        AL_STRATEGY.LIBACT_UNCERTAINTY_SM: "BT (LIB)",
        AL_STRATEGY.LIBACT_UNCERTAINTY_ENT: "Ent (LIB)",
        AL_STRATEGY.SMALLTEXT_LEASTCONFIDENCE: "LC (SM)",
        AL_STRATEGY.SMALLTEXT_PREDICTIONENTROPY: "Ent (SM)",
        AL_STRATEGY.SMALLTEXT_BREAKINGTIES: "BT (SM)",
        AL_STRATEGY.SMALLTEXT_EMBEDDINGKMEANS: "EKM (SM)",
        AL_STRATEGY.SMALLTEXT_DISCRIMINATIVEAL: "DAL (SM)",
        AL_STRATEGY.SMALLTEXT_GREEDYCORESET: "Core (SM)",
        # AL_STRATEGY.SMALLTEXT_LIGHTWEIGHTCORESET: "Lightweight Greedy Coreset (SM)",
        AL_STRATEGY.SMALLTEXT_CONTRASTIVEAL: "CAL (SM)",
        AL_STRATEGY.SMALLTEXT_RANDOM: "Rand (SM)",
        AL_STRATEGY.SKACTIVEML_QBC: "QBC KL (SKA)",
        AL_STRATEGY.SKACTIVEML_US_MARGIN: "MM (SKA)",
        AL_STRATEGY.SKACTIVEML_US_LC: "LC (SKA)",
        AL_STRATEGY.SKACTIVEML_US_ENTROPY: "Ent (SKA)",
        AL_STRATEGY.SKACTIVEML_EXPECTED_AVERAGE_PRECISION: "EAP (SKA)",
        AL_STRATEGY.SKACTIVEML_COST_EMBEDDING: "CE (SKA)",
        AL_STRATEGY.SKACTIVEML_DAL: "DAL (SKA)",
        AL_STRATEGY.SKACTIVEML_MCPAL: "MCPAL (SKA)",
        AL_STRATEGY.SKACTIVEML_QBC_VOTE_ENTROPY: "QBC VE (SM)",
        AL_STRATEGY.SKACTIVEML_QUIRE: "QUIRE (SKA)",
    }

    return renaming_dict[al_strat]
