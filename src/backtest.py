import pandas as pd
import numpy as np
 
 
# ---------- Strategies ----------
 
def avg_rate_every_favorable_day(df, col="Close"):
    """Strategy A: average rate across every day the signal says Favorable."""
    favorable_days = df[df["Value_Signal"] == "Favorable"]
    return favorable_days[col].mean(), favorable_days
 
 
def avg_rate_one_per_episode(df, col="Close"):
    """Strategy B: collapse consecutive Favorable streaks into one 'episode'
    and take the first day of each episode, so a long cheap streak doesn't
    dominate the average just by having more days in it."""
    is_favorable = df["Value_Signal"] == "Favorable"
    state_change = is_favorable != is_favorable.shift()
    episode_id = state_change.cumsum()
 
    tmp = df.copy()
    tmp["_Episode_ID"] = episode_id
    episodes_first_day = tmp[is_favorable].groupby("_Episode_ID", as_index=False).first()
    return episodes_first_day[col].mean(), episodes_first_day
 
 
def _pick_transfer_day(period_df, col="Close"):
    """Helper for Strategy C: first Favorable day in the period if one exists,
    otherwise the period's last day (you still have to pay tuition even if
    the signal never fired). Explicitly captures the real date, since
    groupby(Grouper).apply() can otherwise mislabel the index with the
    group's bin start rather than the row's actual date."""
    favorable = period_df[period_df["Value_Signal"] == "Favorable"]
    row = favorable.iloc[0] if not favorable.empty else period_df.iloc[-1]
    row = row.copy()
    row["Transfer_Date"] = row.name
    return row
 
 
def avg_rate_periodic(df, period_days=90, col="Close"):
    """Strategy C: one simulated transfer per fixed period — the most
    realistic option if you only transfer a few times a year, since it
    forces a decision even in periods where the signal never says Favorable."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transfers = df.groupby(pd.Grouper(freq=f"{period_days}D")).apply(
            lambda g: _pick_transfer_day(g, col=col)
        )
    transfers = transfers.dropna(how="all").set_index("Transfer_Date")
    return transfers[col].mean(), transfers
 
 
# ---------- Baselines ----------
 
def rate_immediate(df, col="Close"):
    """Baseline: transfer on the first eligible day in the dataset."""
    return df[col].iloc[0]
 
 
def rate_random(df, col="Close", n_trials=5000, random_state=42):
    """Baseline: transfer on a random day, averaged over many trials
    (with replacement) rather than trusting one lucky/unlucky draw."""
    samples = df[col].sample(n=n_trials, replace=True, random_state=random_state)
    return samples.mean(), samples
 
 
# ---------- Comparison ----------
 
def build_comparison_table(df, period_days=90, n_trials=5000, random_state=42, col="Close"):
    """Runs all three strategies + both baselines and returns a single
    comparison table with each strategy's edge over the random baseline."""
    avg_a, _ = avg_rate_every_favorable_day(df, col=col)
    avg_b, _ = avg_rate_one_per_episode(df, col=col)
    avg_c, _ = avg_rate_periodic(df, period_days=period_days, col=col)
    avg_immediate = rate_immediate(df, col=col)
    avg_random, _ = rate_random(df, col=col, n_trials=n_trials, random_state=random_state)
 
    results = pd.DataFrame({
        "Strategy": [
            "A: every Favorable day",
            "B: one per episode",
            f"C: one per {period_days}-day period",
            "Immediate transfer",
            "Random day (avg of trials)",
        ],
        "Avg Rate Paid": [avg_a, avg_b, avg_c, avg_immediate, avg_random],
    })
    results["vs Random (%)"] = (
        (results["Avg Rate Paid"] - avg_random) / avg_random * 100
    )
    return results
 
 
# ---------- Validation ----------
 
def check_lookahead_bias(df, percentile_col="Percentile_Rank", value_col="Close",
                          period=180, check_idx=None):
    """Manually recomputes one row's percentile rank using only the raw
    values in its trailing window, and compares it to what's stored.
    A mismatch means the signal is (or was) peeking at future data."""
    if check_idx is None:
        check_idx = len(df) // 2
 
    window = df[value_col].iloc[check_idx - period + 1: check_idx + 1]
    manual_pct = window.rank(pct=True).iloc[-1]
    stored_pct = df[percentile_col].iloc[check_idx]
    match = np.isclose(manual_pct, stored_pct)
 
    return {
        "row_date": df.index[check_idx],
        "manual_percentile": manual_pct,
        "stored_percentile": stored_pct,
        "match": bool(match),
    }
 
 
# ---------- Transaction cost reality check ----------
 
def estimate_roundtrip_spread_pct(gbpusd_spread_pct=0.5, usdidr_spread_pct=1.0):
    """Rough estimate of round-trip cost for a GBP/IDR transfer, built from
    the two component legs (GBP/USD is a major, tighter-spread pair;
    USD/IDR is an emerging-market pair with a wider spread). Replace these
    two numbers with real figures from your actual transfer provider before
    trusting the result."""
    return gbpusd_spread_pct + usdidr_spread_pct
 
 
def check_edge_vs_cost(results, estimated_cost_pct, baseline_label="Random day (avg of trials)"):
    """For each strategy, checks whether its edge over the random baseline
    is bigger than the estimated transaction cost — i.e. whether the timing
    advantage is large enough to survive real-world spread/fees."""
    rows = []
    for _, row in results.iterrows():
        if row["Strategy"] == baseline_label:
            continue
        edge = -row["vs Random (%)"]  # positive = strategy beat random
        survives = edge > estimated_cost_pct
        rows.append({
            "Strategy": row["Strategy"],
            "Edge vs Random (%)": edge,
            "Estimated Cost (%)": estimated_cost_pct,
            "Survives Cost Estimate": survives,
        })
    return pd.DataFrame(rows)