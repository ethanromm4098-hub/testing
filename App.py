"""Warehouse Lease Optimizer: Web Mercantile leasing problem as a Streamlit app.

Linear program
    Decision variables: x[s, L] = square feet leased starting in month s for L months
                        (only leases that end inside the planning horizon)
    Objective:          minimize sum( cost_per_sqft[L] * x[s, L] )
    Constraints:        for every month m, the leases active in month m cover required[m]
                        x[s, L] >= 0, whole square feet
"""

from __future__ import annotations

from dataclasses import dataclass, field

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog

# =============================================================== model ======

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

# ================================================================= app ========

st.set_page_config(page_title="Warehouse Lease Optimizer", page_icon="🏬", layout="wide")

MAX_HORIZON = 12
MAX_COLORED_LEASES = 8  # leases past the eighth fold into "Other leases"

# Categorical colors for the leases, plus chart ink and page surface, per theme.
PALETTE = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                   "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "other": "#898781",
        "ink": "#0b0b0b",
        "surface": "#ffffff",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
                   "#d55181", "#008300", "#9085e9", "#e66767"],
        "other": "#898781",
        "ink": "#ffffff",
        "surface": "#0e1117",
    },
}


def current_palette() -> dict:
    try:
        mode = "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # older Streamlit versions have no theme detection
        mode = "light"
    return PALETTE[mode]


def money(value: float) -> str:
    return f"${value:,.0f}"


def reset_inputs() -> None:
    for key in [k for k in st.session_state if k.startswith(("req_", "cost_"))]:
        del st.session_state[key]
    st.session_state.horizon = len(TEXTBOOK_REQUIREMENTS)


# ---------------------------------------------------------------- inputs ----
if "horizon" not in st.session_state:
    st.session_state.horizon = len(TEXTBOOK_REQUIREMENTS)

with st.sidebar:
    st.header("Inputs")
    horizon = int(
        st.number_input("Planning horizon (months)", min_value=1, max_value=MAX_HORIZON,
                        step=1, key="horizon")
    )

    st.subheader("Space required")
    req_default = pd.DataFrame(
        {
            "Month": list(range(1, horizon + 1)),
            "Required sq ft": [
                TEXTBOOK_REQUIREMENTS[m - 1] if m <= len(TEXTBOOK_REQUIREMENTS) else 0
                for m in range(1, horizon + 1)
            ],
        }
    )
    req_df = st.data_editor(
        req_default,
        key=f"req_{horizon}",
        hide_index=True,
        disabled=["Month"],
        column_config={
            "Month": st.column_config.NumberColumn(format="%d"),
            "Required sq ft": st.column_config.NumberColumn(
                min_value=0, step=1000, format="%d", required=True
            ),
        },
    )

    st.subheader("Lease cost")
    cost_default = pd.DataFrame(
        {
            "Lease length (months)": list(range(1, horizon + 1)),
            "Cost per sq ft": pd.Series(
                [TEXTBOOK_COSTS.get(n) for n in range(1, horizon + 1)], dtype="float"
            ),
        }
    )
    cost_df = st.data_editor(
        cost_default,
        key=f"cost_{horizon}",
        hide_index=True,
        disabled=["Lease length (months)"],
        column_config={
            "Lease length (months)": st.column_config.NumberColumn(format="%d"),
            "Cost per sq ft": st.column_config.NumberColumn(
                min_value=0.0, step=0.01, format="$%.2f"
            ),
        },
    )
    st.caption("Leave a cost blank if that lease length isn't offered.")

    st.button("Reset to case data", on_click=reset_inputs)

# -------------------------------------------------------------- validate ----
st.title("Warehouse Lease Optimizer")
st.write(
    "Finds the cheapest combination of warehouse leases that covers the space "
    "required each month. Loaded with the Web Mercantile case; edit the inputs "
    "in the sidebar and the plan re-solves."
)

requirements = req_df["Required sq ft"]
if requirements.isna().any():
    st.error("Enter the required space for every month.")
    st.stop()

costs = {
    int(length): float(cost)
    for length, cost in zip(cost_df["Lease length (months)"], cost_df["Cost per sq ft"])
    if pd.notna(cost)
}
if not costs:
    st.error("Enter a cost for at least one lease length.")
    st.stop()

req_values = tuple(int(v) for v in requirements)
solution = solve_leasing(list(req_values), costs)
if not solution.ok:
    st.error(solution.message)
    st.stop()

# --------------------------------------------------------------- results ----
baselines = baseline_costs(list(req_values), costs)
total = solution.total_cost

col_opt, col_m2m, col_peak = st.columns(3)
col_opt.metric("Optimal total leasing cost", money(total))

for col, key, label in (
    (col_m2m, "month_to_month", "Lease month-to-month"),
    (col_peak, "peak_all_months", f"Lease the peak for all {horizon} months"),
):
    value = baselines[key]
    if value is None:
        col.metric(label, "n/a", help="Needs a cost for that lease length.")
    else:
        extra = value - total
        col.metric(
            label,
            money(value),
            delta=f"+{money(extra)} vs. optimal" if extra > 0 else "Same as optimal",
            delta_color="inverse" if extra > 0 else "off",
        )

lease_rows = solution.leases.reset_index(drop=True)
if lease_rows.empty:
    st.info("No space is required in any month, so no leases are needed.")
    st.stop()

# Chart: stacked bars show each lease in effect; tick marks show the requirement.
colors = current_palette()
labels = []
bar_rows = []
for i, lease in lease_rows.iterrows():
    label = lease["Lease"] if i < MAX_COLORED_LEASES else "Other leases"
    if label not in labels:
        labels.append(label)
    for month in range(lease["Starts"], lease["Ends"] + 1):
        bar_rows.append(
            {"Month": month, "Lease": label, "Square feet": int(lease["Square feet"]),
             "Order": labels.index(label)}
        )
bar_df = pd.DataFrame(bar_rows).groupby(["Month", "Lease", "Order"], as_index=False).sum()
color_range = [colors["series"][i] if lab != "Other leases" else colors["other"]
               for i, lab in enumerate(labels)]

month_axis = alt.X("Month:O", axis=alt.Axis(labelAngle=0, title="Month"),
                   scale=alt.Scale(domain=list(range(1, horizon + 1))))

bars = (
    alt.Chart(bar_df)
    .mark_bar(stroke=colors["surface"], strokeWidth=2)
    .encode(
        x=month_axis,
        y=alt.Y("Square feet:Q", stack="zero", title="Square feet",
                axis=alt.Axis(format="~s")),
        color=alt.Color("Lease:N", scale=alt.Scale(domain=labels, range=color_range),
                        legend=alt.Legend(title="Lease", orient="bottom")),
        order=alt.Order("Order:Q"),
        tooltip=[alt.Tooltip("Lease:N"), alt.Tooltip("Month:O"),
                 alt.Tooltip("Square feet:Q", format=",")],
    )
)
ticks = (
    alt.Chart(solution.coverage)
    .mark_tick(color=colors["ink"], thickness=3, size=40)
    .encode(
        x=month_axis,
        y=alt.Y("Required:Q", title="Square feet"),
        tooltip=[alt.Tooltip("Month:O"),
                 alt.Tooltip("Required:Q", title="Required sq ft", format=","),
                 alt.Tooltip("Leased:Q", title="Leased sq ft", format=","),
                 alt.Tooltip("Unused:Q", title="Unused sq ft", format=",")],
    )
)

st.subheader("Space leased each month")
st.altair_chart(alt.layer(bars, ticks).properties(height=360))
st.caption("Each color is one lease. The tick mark on each bar shows the space required that month.")

col_plan, col_cover = st.columns([3, 2])
with col_plan:
    st.subheader("Lease plan")
    st.dataframe(
        lease_rows.style.format(
            {"Square feet": "{:,}", "Cost per sq ft": "${:,.2f}", "Lease cost": "${:,.0f}"}
        ),
        hide_index=True,
    )
    st.caption("Several plans can tie for the lowest cost; this is one of them.")
with col_cover:
    st.subheader("Coverage by month")
    st.dataframe(
        solution.coverage.style.format({c: "{:,}" for c in ("Required", "Leased", "Unused")}),
        hide_index=True,
    )

with st.expander("How the model works"):
    st.markdown(
        f"Each decision variable is the square footage of one possible lease, "
        f"defined by its start month *s* and length *L*. Only leases that end by "
        f"month {horizon} are allowed, which gives **{solution.n_options} lease options** "
        f"for these inputs."
    )
    st.latex(r"\min \sum_{s,L} c_L \, x_{s,L}")
    st.latex(
        r"\text{s.t.} \sum_{(s,L)\,:\, s \le m \le s+L-1} x_{s,L} \;\ge\; r_m "
        r"\quad \text{for each month } m"
    )
    st.latex(r"x_{s,L} \ge 0 \text{ and whole square feet}")
    st.markdown(
        "Here $c_L$ is the cost per square foot of an *L*-month lease and $r_m$ is the "
        "space required in month *m*. The model is solved with the HiGHS solver in SciPy."
    )
