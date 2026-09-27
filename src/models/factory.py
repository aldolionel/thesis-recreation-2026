"""Model factory functions, kept pluggable for future model additions."""

from sklearn.ensemble import HistGradientBoostingClassifier

from src.utils.config import SEED


def make_gbm(**kwargs) -> HistGradientBoostingClassifier:
    """Build a seeded gradient boosting classifier.

    Args:
        **kwargs: Keyword arguments forwarded to
            ``HistGradientBoostingClassifier``, overriding the defaults.

    Returns:
        A ``HistGradientBoostingClassifier`` with a fixed random seed and
        ``max_iter=200`` unless overridden.
    """
    params = {"random_state": SEED, "max_iter": 200}
    params.update(kwargs)
    return HistGradientBoostingClassifier(**params)
