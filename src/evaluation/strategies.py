"""Registry of imbalance-handling strategies for the honest evaluation study.

Each strategy declares the feature representation it needs (one-hot for
everything except SMOTENC, which needs integer-coded categoricals) and how
it is applied: an imblearn resampler fit on the training fold only, a
``class_weight`` flag passed to the model, or nothing (``none``).
"""

from collections.abc import Callable
from dataclasses import dataclass

from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from src.utils.config import SEED


@dataclass(frozen=True)
class StrategySpec:
    """Declares how one imbalance-handling strategy is applied.

    Attributes:
        name: Strategy identifier, matches the registry key.
        representation: ``"onehot"`` or ``"coded"`` feature representation
            this strategy needs.
        build_resampler: Factory returning a fitted-ready imblearn resampler,
            or ``None`` if this strategy does not resample. Accepts an
            optional ``cat_idx`` keyword for representation-aware resamplers.
        class_weight: Value passed as ``class_weight`` to the model, or
            ``None``.
        requires_finite_input: Whether this strategy's resampler needs NaN
            imputed first (distance-based resamplers such as SMOTE/ADASYN/
            SMOTENC cannot accept missing values; row-level resamplers such
            as random over/under-sampling can).
    """

    name: str
    representation: str
    build_resampler: Callable[..., object] | None
    class_weight: str | None = None
    requires_finite_input: bool = False


def _build_random_over(**_kwargs) -> RandomOverSampler:
    return RandomOverSampler(random_state=SEED)


def _build_smote(**_kwargs) -> SMOTE:
    return SMOTE(random_state=SEED)


def _build_adasyn(**_kwargs) -> ADASYN:
    return ADASYN(random_state=SEED)


def _build_smotenc(cat_idx: list[int] | None = None, **_kwargs) -> SMOTENC:
    if cat_idx is None:
        raise ValueError("SMOTENC requires cat_idx (categorical feature positions).")
    return SMOTENC(categorical_features=cat_idx, random_state=SEED)


def _build_random_under(**_kwargs) -> RandomUnderSampler:
    return RandomUnderSampler(random_state=SEED)


STRATEGIES: dict[str, StrategySpec] = {
    "none": StrategySpec("none", "onehot", None),
    "class_weight": StrategySpec("class_weight", "onehot", None, class_weight="balanced"),
    "random_over": StrategySpec("random_over", "onehot", _build_random_over),
    "smote": StrategySpec("smote", "onehot", _build_smote, requires_finite_input=True),
    "adasyn": StrategySpec("adasyn", "onehot", _build_adasyn, requires_finite_input=True),
    "smotenc": StrategySpec("smotenc", "coded", _build_smotenc, requires_finite_input=True),
    "random_under": StrategySpec("random_under", "onehot", _build_random_under),
}
