"""Hyperparameter tuning of a support vector machine in host-loop mode.

The objective trains and cross-validates a classifier with scikit-learn, so
it has to run on the CPU. ``mode="host"`` copies the population to the host
for evaluation; with ``vectorized=False`` the objective receives one
candidate at a time. Only the selection and the update rules run on the GPU,
and they are a negligible part of the runtime here: expect no speedup over a
CPU optimizer for a problem like this. The point of the example is that such
objectives can be used at all, with mixed variable types.

Variables: log10 of C, log10 of gamma, and the kernel as a categorical
variable. Objective: one minus the mean accuracy of a stratified three-fold
cross-validation on the breast cancer data set of scikit-learn.

Reference: the accuracy of the classifier with scikit-learn's default
hyperparameters under the same cross-validation, printed by the example.
"""

from __future__ import annotations

import numpy as np
from common import parse_arguments, report, settings

import pygpumeta
from pygpumeta.space import Categorical, Real

KERNELS = ("rbf", "poly", "sigmoid")
SPACE = [Real(-2.0, 3.0), Real(-5.0, 1.0), Categorical(KERNELS)]


def _data():
    from sklearn.datasets import load_breast_cancer

    return load_breast_cancer(return_X_y=True)


def cv_error(log_c: float | None, log_gamma: float | None, kernel: str) -> float:
    """One minus the cross-validated accuracy; None selects the defaults."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    x, y = _data()
    if log_c is None or log_gamma is None:
        model = SVC(kernel=kernel)
    else:
        model = SVC(C=10.0**log_c, gamma=10.0**log_gamma, kernel=kernel)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    scores = cross_val_score(make_pipeline(StandardScaler(), model), x, y, cv=folds)
    return 1.0 - float(scores.mean())


def objective(candidate: np.ndarray) -> float:
    """Evaluate one decoded candidate: (log10 C, log10 gamma, kernel index)."""
    return cv_error(candidate[0], candidate[1], KERNELS[int(candidate[2])])


def run(**options):
    options = {
        "algorithm": "gwo",
        "selection": "tournament",
        "pop_size": 32,
        "max_iter": 12,
        **options,
    }
    options["mode"] = "host"
    return pygpumeta.minimize(objective, SPACE, vectorized=False, **options)


def main() -> None:
    args = parse_arguments(__doc__.splitlines()[0])
    options = settings(args, pop_size=32, max_iter=12)
    if args.quick:
        options.update(pop_size=8, max_iter=3)
    result = run(**options)
    log_c, log_gamma, kernel = result.values
    report(
        "SVM tuning in host-loop mode",
        result,
        {
            "cross-validated accuracy": f"{1.0 - result.fun:.4f}",
            "accuracy with default hyperparameters": f"{1.0 - cv_error(None, None, 'rbf'):.4f}",
            "best hyperparameters": f"C = {10.0**log_c:.4g}, gamma = {10.0**log_gamma:.4g}, "
            f"kernel = {kernel}",
        },
    )


if __name__ == "__main__":
    main()
