import seaborn as sns
from matplotlib import pyplot as plt

from resources.data_types import AL_STRATEGY, LEARNER_MODEL


def set_seaborn_style(font_size=5.8):
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "Times New Roman",
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
    sns.set_style("ticks")
    sns.set_context("paper")
    # sns.set_palette("colorblind")
    # plt.style.use("seaborn-v0_8-paper")
    # plt.style.use("ggplot")
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
        AL_STRATEGY.ALIPY_UNCERTAINTY_LC: "UNC_LC (ALI)",
        AL_STRATEGY.ALIPY_GRAPH_DENSITY: "GD (ALI)",
        AL_STRATEGY.ALIPY_CORESET_GREEDY: "Core (ALI)",
        AL_STRATEGY.OPTIMAL_GREEDY_10: "OG10",
        AL_STRATEGY.OPTIMAL_GREEDY_20: "OG20",
        AL_STRATEGY.LIBACT_UNCERTAINTY_LC: "UNC_LC (LIB)",
        AL_STRATEGY.LIBACT_DWUS: "DWUS (LIB)",
        AL_STRATEGY.LIBACT_QUIRE: "QUIRE (LIB)",
        AL_STRATEGY.ALIPY_DENSITY_WEIGHTED: "DWUS (ALI)",
        AL_STRATEGY.ALIPY_UNCERTAINTY_MM: "UNC_MM (ALI)",
        AL_STRATEGY.ALIPY_UNCERTAINTY_ENTROPY: "UNC_Ent (ALI)",
        AL_STRATEGY.LIBACT_UNCERTAINTY_SM: "UNC_SM (LIB)",
        AL_STRATEGY.LIBACT_UNCERTAINTY_ENT: "UNC_Ent (LIB)",
        AL_STRATEGY.SMALLTEXT_LEASTCONFIDENCE: "UNC_LC (SM)",
        AL_STRATEGY.SMALLTEXT_PREDICTIONENTROPY: "UNC_Ent (SM)",
        AL_STRATEGY.SMALLTEXT_BREAKINGTIES: "UNC_SM (SM)",
        AL_STRATEGY.SMALLTEXT_EMBEDDINGKMEANS: "EKM (SM)",
        AL_STRATEGY.SMALLTEXT_DISCRIMINATIVEAL: "DAL (SM)",
        AL_STRATEGY.SMALLTEXT_GREEDYCORESET: "Core (SM)",
        AL_STRATEGY.SMALLTEXT_LIGHTWEIGHTCORESET: "Lightweight Greedy Coreset (SM)",
        AL_STRATEGY.SMALLTEXT_CONTRASTIVEAL: "CAL (SM)",
        AL_STRATEGY.SMALLTEXT_RANDOM: "Rand (SM)",
        AL_STRATEGY.SKACTIVEML_QBC: "QBC KL (SKA)",
        AL_STRATEGY.SKACTIVEML_US_MARGIN: "UNC_MM (SKA)",
        AL_STRATEGY.SKACTIVEML_US_LC: "UNC_LC (SKA)",
        AL_STRATEGY.SKACTIVEML_US_ENTROPY: "UNC_Ent (SKA)",
        AL_STRATEGY.SKACTIVEML_EXPECTED_AVERAGE_PRECISION: "EAP (SKA)",
        AL_STRATEGY.SKACTIVEML_COST_EMBEDDING: "CE (SKA)",
        AL_STRATEGY.SKACTIVEML_DAL: "DAL (SKA)",
        AL_STRATEGY.SKACTIVEML_MCPAL: "MCPAL (SKA)",
        AL_STRATEGY.SKACTIVEML_QBC_VOTE_ENTROPY: "QBC VE (SM)",
        AL_STRATEGY.SKACTIVEML_QUIRE: "QUIRE (SKA)",
    }

    return renaming_dict[al_strat]


def _rename_learner_model(learner_model: str) -> str:
    if learner_model == "Gold Standard":
        return "Gold Standard"
    learner_model = LEARNER_MODEL(int(learner_model))

    renaming_dict = {
        LEARNER_MODEL.MLP: "MLP (1 layer)",
        LEARNER_MODEL.RBF_SVM: "SVM (RBF)",
        LEARNER_MODEL.RF: "RF",
        LEARNER_MODEL.DEEP_LEARNING2: "MLP (3 layer)",
        LEARNER_MODEL.DEEP_LEARNING3: "MLP (6 layer)",
    }

    return renaming_dict[learner_model]
