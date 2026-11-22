# Adding an algorithm

An algorithm that can be written as array operations plus a selection needs
no CUDA code. There are two ways to write one.

## A plain loop

`examples/08_custom_algorithm.py` is a complete evolution strategy in about
twenty lines. The population is a CuPy array, the objective is evaluated
with array operations, and the parents come from a selection operator:

```python
import cupy as cp
from pygpumeta import selection

values = objective(population)  # cupy array, shape (n,)
for it in range(max_iter):
    parents = selection.tournament(values, size=3, seed=seed + it)
    children = population[parents] + width * rng.standard_normal(population.shape)
    child_values = objective(children)
    better = child_values < values
    population = cp.where(better[:, None], children, population)
    values = cp.where(better, child_values, values)
```

Give every call a different seed, as above, or every iteration draws the
same competitors. `pygpumeta.reference` has the same functions for NumPy
arrays and returns the same indices, which is convenient for developing and
testing on a machine without a GPU.

## The way PSO and DE are written

`PSO` and `DE` in `pygpumeta/algorithms/pso.py` and `de.py` are written on
a small base class that supplies bounds, the objective modes, constraints,
decoding of integer variables, the best-so-far bookkeeping, stopping rules
and the result object. An algorithm implements two methods. This is all of
DE's iteration:

```python
def _iterate(self, it):
    xp, ps, dim = self._xp, self._ps, self.dim
    own = xp.arange(ps)
    selected = self._select(self._scores, it)  # None for selection="native"
    offsets = 1 + (self._rng.random_sample((3, ps)) * (ps - 1)).astype(xp.int64)
    partners = (own + offsets) % ps
    base = self._x[partners[0]] if selected is None else self._x[selected]
    mutant = base + self._weight * (self._x[partners[1]] - self._x[partners[2]])
    take = self._rng.random_sample((ps, dim)) < self._crossover
    take[own, (self._rng.random_sample(ps) * dim).astype(xp.int64)] = True
    trial = self._clip(xp.where(take, mutant, self._x))
    trial_scores = self._score(trial)
    accept = trial_scores <= self._scores
    self._x = xp.where(accept[:, None], trial, self._x)
    self._scores = xp.where(accept, trial_scores, self._scores)
    self._update_best(self._x, self._scores)
```

The helpers: `self._xp` is `cupy` or `numpy`, `self._rng` a seeded random
state of that module, `self._score` evaluates positions (objective, sign,
decoding and constraints included), `self._select` runs the chosen
selection operator, `self._clip` projects onto the bounds and
`self._update_best` records the best position.

This base class is internal: its names start with an underscore and may
change. If you write an algorithm on it, pin the version of pygpumeta
you developed against, or copy the loop style of the first approach.
