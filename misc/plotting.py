import seaborn as sns

from matplotlib import pyplot as plt


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


# https://jwalton.info/Embed-Publication-Matplotlib-Latex/
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
        fig_height_in *= 0.7
    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
