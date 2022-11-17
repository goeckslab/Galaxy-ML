from scipy.stats import spearmanr

from sklearn.metrics import make_scorer


def spearman_correlation_score(y_true, y_pred):
    """Spearman's rank correlation coefficient

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    corr : float
        Spearman rank correlation coefficient
    """
    corr, _ = spearmanr(y_true, y_pred)

    return corr


spearman_correlation_scorer = make_scorer(spearman_correlation_score)
