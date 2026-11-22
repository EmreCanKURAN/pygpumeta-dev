"""Solar cell parameter extraction with the single-diode model.

The current of a solar cell at voltage V is modelled as

    I = Iph - I0 (exp((V + I Rs) / (n Vt)) - 1) - (V + I Rs) / Rsh.

The five parameters (photocurrent Iph, saturation current I0, series and
shunt resistance Rs and Rsh, ideality factor n) are fitted to the 26
measured points of the 57 mm RTC France silicon cell at 33 degrees Celsius
(Easwarakhantan et al., Int. J. Solar Energy 4, 1986) by minimising the
root mean square error of the current.

Reference: the lowest root mean square error reported for this data set
and these bounds is 9.8602e-4 A, a value many parameter-extraction papers
arrive at, with Iph = 0.7608 A, I0 = 0.3230 uA, Rs = 0.0364 Ohm,
Rsh = 53.72 Ohm and n = 1.4812. Evaluating the model at these parameters
reproduces that error.
"""

from __future__ import annotations

import numpy as np
from common import array_module, parse_arguments, report, settings

import pygpumeta

VOLTAGE = np.array([
    -0.2057, -0.1291, -0.0588, 0.0057, 0.0646, 0.1185, 0.1678, 0.2132, 0.2545, 0.2924, 0.3269,
    0.3585, 0.3873, 0.4137, 0.4373, 0.4590, 0.4784, 0.4960, 0.5119, 0.5265, 0.5398, 0.5521,
    0.5633, 0.5736, 0.5833, 0.5900,
])  # fmt: skip
CURRENT = np.array([
    0.7640, 0.7620, 0.7605, 0.7605, 0.7600, 0.7590, 0.7570, 0.7570, 0.7555, 0.7540, 0.7505,
    0.7465, 0.7385, 0.7280, 0.7065, 0.6755, 0.6320, 0.5730, 0.4990, 0.4130, 0.3165, 0.2120,
    0.1035, -0.0100, -0.1230, -0.2100,
])  # fmt: skip
THERMAL_VOLTAGE = 1.3806503e-23 * (273.15 + 33.0) / 1.60217646e-19
# Iph [A], I0 [uA], Rs [Ohm], Rsh [Ohm], n. I0 is searched in microampere so
# that all variables have a comparable scale.
BOUNDS = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5), (0.0, 100.0), (1.0, 2.0)]
REFERENCE_RMSE = 9.8602e-4


def rmse(parameters):
    """Root mean square error of the modelled current; one value per row."""
    xp = array_module(parameters)
    v, i = xp.asarray(VOLTAGE), xp.asarray(CURRENT)
    iph, i0, rs, rsh, n = (parameters[:, k : k + 1] for k in range(5))
    drop = v + i * rs
    # The measured current is used on the right-hand side, which is the
    # customary explicit form of this benchmark.
    exponent = xp.minimum(drop / (n * THERMAL_VOLTAGE), 700.0)
    model = iph - 1e-6 * i0 * (xp.exp(exponent) - 1.0) - drop / xp.maximum(rsh, 1e-9)
    return xp.sqrt(((model - i) ** 2).mean(axis=1))


def run(**options):
    options = {
        "algorithm": "de",
        "selection": "native",
        "pop_size": 1024,
        "max_iter": 3000,
        **options,
    }
    return pygpumeta.minimize(rmse, BOUNDS, **options)


def main() -> None:
    args = parse_arguments(__doc__.splitlines()[0])
    result = run(**settings(args, pop_size=1024, max_iter=3000))
    iph, i0, rs, rsh, n = result.x
    report(
        "Solar cell: single-diode model, RTC France data",
        result,
        {
            "root mean square error": f"{result.fun:.6e} A",
            "best value in the literature": f"{REFERENCE_RMSE:.4e} A",
            "parameters": f"Iph = {iph:.5f} A, I0 = {i0:.5f} uA, Rs = {rs:.5f} Ohm, "
            f"Rsh = {rsh:.3f} Ohm, n = {n:.5f}",
        },
    )


if __name__ == "__main__":
    main()
