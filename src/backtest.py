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
 
 
# ---------- Walk-forward evaluation ----------

def walk_forward_folds(df, n_folds=5):
    """Split df into n_folds sequential, non-overlapping chronological folds
    (no shuffling) -- e.g. successive multi-year blocks rather than random
    samples, so each fold lines up with a real stretch of market history."""
    n = len(df)
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n  # last fold takes the remainder
        folds.append(df.iloc[start:end])
    return folds


def walk_forward_evaluate(df, n_folds=5, period_days=90, n_trials=5000,
                           random_state=42, col="Close"):
    """Runs build_comparison_table independently on each of n_folds sequential
    time windows instead of once over the whole history, and reports the
    spread of each strategy's edge over the random baseline across folds.

    This is the rule-based-signal counterpart to the ML walk-forward in
    notebook 04: same idea (many sequential windows, look at the spread
    instead of trusting one pooled number), applied here to Strategies A/B/C.
    There's no train/test purge gap like the ML version needs -- these
    strategies don't fit anything to the data, they just read the rolling
    percentile signal, which is already causal within each fold.

    A strategy whose edge only looks good pooled across every year (mixing
    GBP/IDR's balanced and trending stretches together) but is inconsistent,
    or flips sign, fold-to-fold isn't something to trust.

    Returns (fold_results, summary):
      fold_results -- one row per (fold, strategy), with the fold's date
                       range, avg rate paid, and edge vs. that fold's own
                       random baseline.
      summary       -- one row per strategy: mean/std edge across folds and
                        how many of the n_folds it beat the random baseline in.
    """
    folds = walk_forward_folds(df, n_folds=n_folds)

    rows = []
    for fold_num, fold_df in enumerate(folds, start=1):
        table = build_comparison_table(
            fold_df, period_days=period_days, n_trials=n_trials,
            random_state=random_state, col=col,
        )
        table = table[table["Strategy"] != "Random day (avg of trials)"].copy()
        table["Fold"] = fold_num
        table["Start"] = fold_df.index.min()
        table["End"] = fold_df.index.max()
        table["Edge vs Random (%)"] = -table["vs Random (%)"]  # positive = beat random
        rows.append(table)

    fold_results = pd.concat(rows, ignore_index=True)
    fold_results = fold_results[
        ["Fold", "Start", "End", "Strategy", "Avg Rate Paid", "Edge vs Random (%)"]
    ]

    summary = fold_results.groupby("Strategy")["Edge vs Random (%)"].agg(
        mean_edge="mean", std_edge="std",
        folds_beating_random=lambda s: (s > 0).sum(),
    ).reset_index()
    summary["n_folds"] = n_folds

    return fold_results, summary


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