# The selection operators

The library contains two selection operators that exchange objective values
between the 32 threads of a warp through registers, without shared or
global memory traffic, and a third that adds an occasional comparison with
the global best. They are described in

> E. C. Kuran, U. Kuran, E. Ecemiş, "Register-Level Selection via Warp
> Primitives: Shuffle-Based Tournament and Ballot-Elite Mechanisms for
> GPU-Accelerated Metaheuristics," Swarm and Evolutionary Computation, 2026.
> https://doi.org/10.1016/j.swevo.2026.102633

| Operator | In `minimize` | Standalone | What an individual receives |
| --- | --- | --- | --- |
| Warp tournament | `"tournament"`, `Tournament(size)` | `selection.tournament` | the best of itself and `size` members of its warp |
| Ballot-Elite | `"ballot_elite"`, `BallotElite(size, alpha)` | `selection.ballot_elite` | an elite of its warp, where elite means fitness above `alpha` times the mean fitness; a tournament if the warp has no elite |
| Migration | `"migration"`, `Migration(rate, size)` | `selection.migration` | the tournament winner, compared with the global best with probability `rate` |
| Native | `"native"` | | whatever the original algorithm prescribes |

The default, `selection="auto"`, uses the warp tournament for ABC, GWO, WOA
and HHO and the native rule for SCA, SSA, PSO and DE.

## What they provide

- **Selection at register speed for any population size.** The operators
  work on a device array of objective values and return, for every
  individual, the index of a selected individual. Population sizes that are
  not multiples of 32 are handled correctly.
- **Reproducibility.** For a given seed the result is defined exactly, and
  `pygpumeta.reference` reproduces it bit for bit on the CPU. The test
  suite compares the two for population sizes from 1 to 8192.
- **A building block.** They accept PyCUDA and CuPy arrays, so an algorithm
  of your own can use them directly; see
  [Adding an algorithm](adding_an_algorithm.md).

## What they do not provide

- **They are not the source of the speedup over a CPU.** The speedups of
  the GPU implementations over CPU implementations come from running the
  population in parallel on the GPU. Selection is a small part of an
  iteration.
- **They do not improve solution quality in general.** In the study above,
  replacing the native rule by a warp operator improved the results of ABC,
  WOA and HHO, made no difference for SCA, and made the results of SSA
  worse. Treat the operator as a setting to choose per algorithm and
  problem, and compare with `selection="native"` on your own problem using
  the same seeds.
- **They are local.** Tournament and Ballot-Elite compare an individual
  only with individuals whose index shares `index // 32` with its own. Good
  solutions therefore spread through the population over several
  iterations. Migration exists to shorten that.
- **Ballot-Elite above 256 individuals is not guaranteed to be
  bit-reproducible between runs.** Its threshold is a mean computed by an
  atomic sum over thread blocks, whose order is not fixed, so the last bits
  of the mean may differ from run to run and change a selection. Repeated
  runs in the test suite are identical, but the kernel does not promise it.
  Tournament and migration are unaffected.

## Parameters

- `size`, between 1 and 6: the number of competitors drawn. Larger means
  stronger selection pressure.
- `alpha` (Ballot-Elite), default 1.2: individuals with fitness above
  `alpha` times the mean are elite. Fitness is `1 / (1 + f)` for `f >= 0`
  and `1 + |f|` otherwise.
- `rate` (migration), default 0.25: `0` is the plain tournament, `1`
  selects an individual with the best objective value for everyone.

## How the algorithms use a selection

| Algorithm | With an operator | `"native"` |
| --- | --- | --- |
| ABC | onlooker `i` exploits the food source selected for it | roulette wheel on the fitness |
| GWO | the three leaders are the best distinct selected individuals | the three best individuals |
| WOA, SCA, HHO | individual `i` is guided by its selected individual | guided by the best position so far |
| SSA | the food source of leader `i` is its selected individual | the best position so far |
| PSO | particle `i` follows the personal best of its selected particle | the global best |
| DE | the base vector of individual `i` is its selected individual | a random individual |
