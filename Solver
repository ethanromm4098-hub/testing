"""Linear-programming model for the Web Mercantile warehouse-leasing problem.

Decision variables
    x[s, L] = square feet leased starting in month s for L months
              (only leases that end inside the planning horizon).

Objective
    minimize  sum( cost_per_sqft[L] * x[s, L] )

Constraints
    for every month m:  sum of x[s, L] for leases active in month m  >=  required[m]
    x[s, L] >= 0, whole square feet
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog

TEXTBOOK_REQUIREMENTS = [30_000, 20_000, 40_000, 10_000, 50_000]
TEXTBOOK_COSTS = {1: 65, 2: 100, 3: 135, 4: 160, 5: 190}


@dataclass
class LeaseSolution:
    ok: bool
    message: str
    total_cost: float | None = None
    leases: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_options: int = 0


def candidate_leases(horizon: int, costs: dict[int, float]) -> list[tuple[int, int]]:
    """Every (start month, length) lease that is offered and ends inside the horizon."""
    return [
        (start, length)
        for start in range(1, horizon + 1)
        for length in sorted(costs)
        if start + length - 1 <= horizon
    ]


def lease_label(start: int, length: int) -> str:
    end = start + length - 1
    return f"Month {start}" if length == 1 else f"Months {start}–{end}"


def solve_leasing(requirements: list[float], costs: dict[int, float]) -> LeaseSolution:
    """Return the minimum-cost set of leases that covers every month's requirement."""
    horizon = len(requirements)
    options = candidate_leases(horizon, costs)
    if not options:
        return LeaseSolution(False, "No offered lease length fits inside the planning horizon.")

    # Coverage matrix: row m, column j is 1 when lease j is active in month m.
    active = np.zeros((horizon, len(options)))
    for j, (start, length) in enumerate(options):
        active[start - 1 : start - 1 + length, j] = 1

    cost_vector = np.array([costs[length] for _, length in options], dtype=float)
    required = np.asarray(requirements, dtype=float)

    result = linprog(
        c=cost_vector,
        A_ub=-active,  # active @ x >= required  ->  -active @ x <= -required
        b_ub=-required,
        bounds=(0, None),
        integrality=np.ones(len(options)),  # whole square feet
        method="highs",
    )
    if not result.success:
        return LeaseSolution(
            False,
            "No combination of the offered leases covers every month. "
            "Offer more lease lengths (a 1-month lease always works).",
            n_options=len(options),
        )

    x = np.round(result.x).astype(int)

    rows = []
    for (start, length), sqft in zip(options, x):
        if sqft > 0:
            rows.append(
                {
                    "Lease": lease_label(start, length),
                    "Starts": start,
                    "Ends": start + length - 1,
                    "Length (months)": length,
                    "Square feet": sqft,
                    "Cost per sq ft": costs[length],
                    "Lease cost": sqft * costs[length],
                }
            )
    columns = ["Lease", "Starts", "Ends", "Length (months)", "Square feet",
               "Cost per sq ft", "Lease cost"]
    leases = pd.DataFrame(rows, columns=columns).sort_values(
        ["Starts", "Length (months)"], ignore_index=True
    )

    leased = active @ x
    coverage = pd.DataFrame(
        {
            "Month": range(1, horizon + 1),
            "Required": required.astype(int),
            "Leased": leased.astype(int),
            "Unused": (leased - required).astype(int),
        }
    )

    return LeaseSolution(
        ok=True,
        message="Optimal",
        total_cost=float(leases["Lease cost"].sum()),
        leases=leases,
        coverage=coverage,
        n_options=len(options),
    )


def baseline_costs(requirements: list[float], costs: dict[int, float]) -> dict[str, float | None]:
    """Cost of the two simple strategies the case describes, or None if not possible."""
    horizon = len(requirements)
    month_to_month = costs[1] * sum(requirements) if 1 in costs else None
    peak_all_months = costs[horizon] * max(requirements) if horizon in costs else None
    return {"month_to_month": month_to_month, "peak_all_months": peak_all_months}
