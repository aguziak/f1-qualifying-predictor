import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt


def plot_error_dist(errors: pd.Series, plot_z_score: bool = False, error_name: str = 'Error'):
    """
    Creates a histogram and QQ plot for the provided error distribution

    Args:
        errors (Series): Pandas Series object containing error data
        error_name (str): The name of the error being plotting, which will be used for axis labeling
        plot_z_score (bool): If true, will create the histogram using z-scores instead of raw scores

    """

    fig, (ax1, ax2) = plt.subplots(2)

    fig: plt.Figure
    ax1: plt.Axes
    ax2: plt.Axes

    n_bins = 50

    std_dev = np.std(errors)
    avg = np.average(errors)
    z_scores = (errors - avg) / std_dev

    if plot_z_score:
        bin_width = (z_scores.max() - z_scores.min()) / n_bins
        gaussian_x = np.linspace(min(z_scores), max(z_scores), 100)
        gaussian_y = scipy.stats.norm.pdf(gaussian_x, 0, 1)
        gaussian_y *= (len(errors) * bin_width)
        ax1.hist(x=z_scores, edgecolor='k', linewidth=1, bins=n_bins)
        ax1.set_xlabel('Z-Score')
    else:
        bin_width = (errors.max() - errors.min()) / n_bins
        gaussian_x = np.linspace(min(errors), max(errors), 100)
        gaussian_y = scipy.stats.norm.pdf(gaussian_x, avg, std_dev)
        gaussian_y *= (len(errors) * bin_width)
        ax1.hist(x=errors, edgecolor='k', linewidth=1, bins=n_bins)
        ax1.axvline(x=avg, label=f'Mean Value ({avg:.3f})', color='k', linestyle='--')
        ax1.set_xlabel(f'{error_name}')

    ax1.plot(gaussian_x, gaussian_y, color='r', linestyle='--', label='Scaled Normal Curve')
    ax1.set_title('Error Distribution')
    ax1.set_ylabel('Count')
    ax1.legend()

    n = len(errors)
    single_lap_pct_diff_normal_quantiles = scipy.stats.norm.ppf(
        (np.arange(1, n + 1)) / (n + 1),
        0,
        1)
    ax2.scatter(x=single_lap_pct_diff_normal_quantiles, y=z_scores.sort_values())
    ax2.plot(single_lap_pct_diff_normal_quantiles, single_lap_pct_diff_normal_quantiles, linestyle='--', color='k')
    ax2.set_title('QQ Plot')
    ax2.set_xlabel('Normal Theoretical Quantiles')
    ax2.set_ylabel('Observed Quantiles')

    plt.tight_layout()
    plt.show()
