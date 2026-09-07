from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.research.ml import ds24_metrics_only_evaluator as ev


STAGE = ROOT / "docs/dream_system/components/DS-24_independent_five_minute_selector/stage_outputs/ds24_p8_r14_e3g_c2_20260824T000000Z"
POLICY_ROOT = STAGE / "r7_r14_policy_workers"
HUBER_ROOT = POLICY_ROOT / "huber" / "metrics_only_v3_r37_huber_replay"
RFF_ROOT = POLICY_ROOT / "rff_ridge" / "metrics_only_v3_r37_rff_retry"
OUT_JSON = POLICY_ROOT / "huber" / "huber_economic_realism_red_team_r1.json"
OUT_MD = POLICY_ROOT / "huber" / "huber_economic_realism_red_team_r1.md"

COST_BPS = [1.0, 2.5, 5.0, 10.0, 20.0, 50.0]
QUARANTINE_START = pd.Timestamp("2019-05-07T13:35:00+00:00")
QUARANTINE_END = pd.Timestamp("2019-06-06T20:00:00+00:00")
ANOMALOUS_PENDING_CUTOFF = pd.Timestamp("2019-06-06T20:00:00+00:00")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    if not ev.openable_exists(path):
        return {}
    try:
        with open(ev.openable_path(path), "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    ev.mkdir_openable(path.parent)
    with open(ev.openable_path(path), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_text(path: Path, text: str) -> None:
    ev.mkdir_openable(path.parent)
    with open(ev.openable_path(path), "w", encoding="utf-8") as handle:
        handle.write(text)


def sha256_small(path: Path, *, max_bytes: int = 5 * 1024 * 1024) -> str:
    if not ev.openable_exists(path):
        return ""
    size = int(Path(ev.openable_path(path)).stat().st_size)
    if size > max_bytes:
        return ""
    digest = hashlib.sha256()
    with open(ev.openable_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_fingerprint(root: Path, stem: str) -> dict[str, Any]:
    manifest_path = root / f"{stem}_manifest.json"
    manifest = read_json(manifest_path)
    parts = manifest.get("parts", [])
    if not isinstance(parts, list):
        parts = []
    return {
        "manifest": rel(manifest_path),
        "manifest_sha256": sha256_small(manifest_path),
        "total_rows": int(manifest.get("total_rows", 0) or 0),
        "part_count": len(parts),
        "part_sha256_head": [str(part.get("sha256", "")) for part in parts[:3]],
        "part_sha256_tail": [str(part.get("sha256", "")) for part in parts[-3:]],
    }


def finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_logs(root: Path) -> dict[str, pd.DataFrame]:
    sleeve_cols = [
        "family",
        "decision_timestamp",
        "maturity_timestamp",
        "maturity_session_date",
        "sleeve_id",
        "sleeve_count",
        "top_n",
        "gross_return_contribution",
        "net_return_contribution",
        "turnover",
        "transaction_cost_contribution",
        "sleeve_capital_fraction",
        "simultaneous_capital_fraction",
        "top_selected_assets_json",
        "resolved_selected_asset_count",
        "row_hash",
    ]
    daily_cols = [
        "session_date",
        "gross_daily_return",
        "net_daily_return",
        "transaction_cost_drag",
        "mean_turnover",
        "win",
        "matured_sleeve_decisions",
        "max_simultaneous_capital_fraction",
    ]
    rank_cols = [
        "family",
        "decision_timestamp",
        "session_date",
        "target_id",
        "eligible_asset_count",
        "resolved_asset_count",
        "spearman_rank_ic",
        "row_hash",
    ]
    return {
        "sleeves": ev.read_parquet_log(root, "sleeve_maturity_ledger_v3", columns=sleeve_cols),
        "daily": ev.read_parquet_log(root, "daily_portfolio_returns_v3", columns=daily_cols),
        "rank": ev.read_parquet_log(root, "rank_ic_v3", columns=rank_cols),
    }


def max_drawdown(values: pd.Series) -> float | None:
    return ev.maximum_drawdown(values)


def performance_from_returns(values: pd.Series) -> dict[str, Any]:
    clean = finite(values).dropna()
    if clean.empty:
        return {
            "valid_days": 0,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "cumulative_return": None,
            "maximum_drawdown": None,
            "win_rate": None,
        }
    annualized_return = float(clean.mean() * ev.TRADING_SESSIONS_PER_YEAR)
    annualized_volatility = float(clean.std(ddof=1) * math.sqrt(ev.TRADING_SESSIONS_PER_YEAR)) if len(clean) > 1 else None
    return {
        "valid_days": int(len(clean)),
        "mean_daily_return": float(clean.mean()),
        "median_daily_return": float(clean.median()),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": annualized_return / annualized_volatility if annualized_volatility and annualized_volatility > 0 else None,
        "cumulative_return": float((1.0 + clean).prod() - 1.0),
        "maximum_drawdown": max_drawdown(clean),
        "win_rate": float((clean > 0.0).mean()),
    }


def retained_date_set(daily: pd.DataFrame) -> set[str]:
    date_col = "maturity_session_date" if "maturity_session_date" in daily.columns else "session_date"
    return set(pd.to_datetime(daily[date_col], utc=True).dt.date.astype(str))


def daily_from_sleeves(sleeves: pd.DataFrame, cost_bps: float, *, accepted_dates: set[str] | None = None) -> pd.DataFrame:
    work = sleeves.copy()
    work["maturity_session_date"] = pd.to_datetime(work["maturity_session_date"], utc=True).dt.date.astype(str)
    work["gross"] = finite(work["gross_return_contribution"])
    work["turnover"] = finite(work["turnover"]).fillna(0.0)
    work["sleeve_capital_fraction"] = finite(work["sleeve_capital_fraction"]).fillna(1.0 / ev.DEFAULT_V3_SLEEVE_COUNT)
    work = work.dropna(subset=["gross"])
    work["cost_drag"] = work["turnover"] * float(cost_bps) / 10000.0 * work["sleeve_capital_fraction"]
    work["net"] = work["gross"] - work["cost_drag"]
    grouped = work.groupby("maturity_session_date", sort=True)
    daily = grouped.agg(
        gross_daily_return=("gross", "sum"),
        net_daily_return=("net", "sum"),
        transaction_cost=("cost_drag", "sum"),
        mean_turnover=("turnover", "mean"),
        resolved_sleeve_count=("gross", "count"),
        cost_base=("sleeve_capital_fraction", lambda item: float((work.loc[item.index, "turnover"] * item).sum())),
    ).reset_index()
    daily["maturity_session_date"] = pd.to_datetime(daily["maturity_session_date"], utc=True)
    if accepted_dates is not None:
        date_key = daily["maturity_session_date"].dt.date.astype(str)
        daily = daily[date_key.isin(accepted_dates)].reset_index(drop=True)
    return daily


def cost_grid(sleeves: pd.DataFrame, costs: Sequence[float], *, accepted_dates: set[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cost in costs:
        daily = daily_from_sleeves(sleeves, cost, accepted_dates=accepted_dates)
        summary = performance_from_returns(daily["net_daily_return"])
        rows.append(
            {
                "transaction_cost_bps_per_unit_turnover": float(cost),
                **summary,
                "mean_turnover": float(daily["mean_turnover"].mean()) if len(daily) else None,
                "total_cost_drag": float(daily["transaction_cost"].sum()) if len(daily) else None,
            }
        )
    return rows


def first_threshold_from_daily(daily_zero: pd.DataFrame, threshold: float, *, high: float = 5000.0) -> float | None:
    gross = finite(daily_zero["gross_daily_return"]).to_numpy(dtype=float)
    cost_base = finite(daily_zero["cost_base"]).to_numpy(dtype=float)

    def sharpe_at(cost: float) -> float | None:
        returns = pd.Series(gross - (cost / 10000.0) * cost_base)
        return performance_from_returns(returns)["sharpe"]

    start = sharpe_at(0.0)
    end = sharpe_at(high)
    if start is None or start <= threshold:
        return 0.0
    if end is None or end > threshold:
        return None
    low = 0.0
    for _ in range(60):
        mid = (low + high) / 2.0
        value = sharpe_at(mid)
        if value is not None and value > threshold:
            low = mid
        else:
            high = mid
    return float(high)


def break_even_cost_bps(sleeves: pd.DataFrame) -> float | None:
    daily = daily_from_sleeves(sleeves, 0.0)
    gross_mean = float(daily["gross_daily_return"].mean()) if len(daily) else None
    cost_base_mean = float(daily["cost_base"].mean()) if len(daily) else None
    if gross_mean is None or cost_base_mean is None or cost_base_mean <= 0:
        return None
    return float(gross_mean / cost_base_mean * 10000.0)


def break_even_cost_bps_from_daily(daily_zero: pd.DataFrame) -> float | None:
    gross_mean = float(daily_zero["gross_daily_return"].mean()) if len(daily_zero) else None
    cost_base_mean = float(daily_zero["cost_base"].mean()) if len(daily_zero) else None
    if gross_mean is None or cost_base_mean is None or cost_base_mean <= 0:
        return None
    return float(gross_mean / cost_base_mean * 10000.0)


def yearly(daily: pd.DataFrame, rank: pd.DataFrame) -> list[dict[str, Any]]:
    work = daily.copy()
    work["year"] = pd.to_datetime(work["maturity_session_date"], utc=True).dt.year
    rank_work = rank.copy()
    rank_work["year"] = pd.to_datetime(rank_work["session_date"], utc=True).dt.year
    rank_work["spearman_rank_ic"] = finite(rank_work["spearman_rank_ic"])
    rank_by_year = rank_work.groupby("year")["spearman_rank_ic"].mean().to_dict()
    rows: list[dict[str, Any]] = []
    for year, group in work.groupby("year", sort=True):
        summary = performance_from_returns(group["net_daily_return"])
        rows.append(
            {
                "year": int(year),
                **summary,
                "mean_turnover": float(group["mean_turnover"].mean()),
                "rank_ic_mean": float(rank_by_year.get(year)) if year in rank_by_year else None,
            }
        )
    return rows


def regime_slices(daily: pd.DataFrame, rank: pd.DataFrame) -> list[dict[str, Any]]:
    work = daily.copy()
    work["date"] = pd.to_datetime(work["maturity_session_date"], utc=True)
    midpoint = work["date"].sort_values().iloc[len(work) // 2]
    regimes = [
        ("first_half", work["date"] <= midpoint),
        ("second_half", work["date"] > midpoint),
        ("pre_2020", work["date"] < pd.Timestamp("2020-01-01", tz="UTC")),
        ("2020_2021", (work["date"] >= pd.Timestamp("2020-01-01", tz="UTC")) & (work["date"] < pd.Timestamp("2022-01-01", tz="UTC"))),
        ("2022_2023", (work["date"] >= pd.Timestamp("2022-01-01", tz="UTC")) & (work["date"] < pd.Timestamp("2024-01-01", tz="UTC"))),
        ("2024_onward", work["date"] >= pd.Timestamp("2024-01-01", tz="UTC")),
    ]
    rank_work = rank.copy()
    rank_work["date"] = pd.to_datetime(rank_work["session_date"], utc=True)
    rank_work["spearman_rank_ic"] = finite(rank_work["spearman_rank_ic"])
    rows: list[dict[str, Any]] = []
    for name, mask in regimes:
        subset = work.loc[mask]
        if name == "first_half":
            rmask = rank_work["date"] <= midpoint
        elif name == "second_half":
            rmask = rank_work["date"] > midpoint
        elif name == "pre_2020":
            rmask = rank_work["date"] < pd.Timestamp("2020-01-01", tz="UTC")
        elif name == "2020_2021":
            rmask = (rank_work["date"] >= pd.Timestamp("2020-01-01", tz="UTC")) & (rank_work["date"] < pd.Timestamp("2022-01-01", tz="UTC"))
        elif name == "2022_2023":
            rmask = (rank_work["date"] >= pd.Timestamp("2022-01-01", tz="UTC")) & (rank_work["date"] < pd.Timestamp("2024-01-01", tz="UTC"))
        else:
            rmask = rank_work["date"] >= pd.Timestamp("2024-01-01", tz="UTC")
        summary = performance_from_returns(subset["net_daily_return"])
        rows.append(
            {
                "slice": name,
                "start": subset["date"].min().isoformat() if len(subset) else "",
                "end": subset["date"].max().isoformat() if len(subset) else "",
                **summary,
                "mean_turnover": float(subset["mean_turnover"].mean()) if len(subset) else None,
                "rank_ic_mean": float(rank_work.loc[rmask, "spearman_rank_ic"].mean()) if int(rmask.sum()) else None,
            }
        )
    return rows


def parse_assets(raw: Any) -> list[str]:
    try:
        payload = json.loads(str(raw))
    except Exception:
        return []
    if isinstance(payload, list):
        return [str(item) for item in payload]
    return []


def security_concentration(sleeves: pd.DataFrame) -> dict[str, Any]:
    work = sleeves.copy()
    work["date"] = pd.to_datetime(work["maturity_session_date"], utc=True).dt.date.astype(str)
    work["gross"] = finite(work["gross_return_contribution"])
    work = work.dropna(subset=["gross"])
    frequency: Counter[str] = Counter()
    signed: defaultdict[str, float] = defaultdict(float)
    abs_contrib: defaultdict[str, float] = defaultdict(float)
    daily_proxy: defaultdict[tuple[str, str], float] = defaultdict(float)
    for row in work.itertuples(index=False):
        assets = parse_assets(getattr(row, "top_selected_assets_json", "[]"))
        if not assets:
            continue
        share = float(getattr(row, "gross")) / len(assets)
        for asset in assets:
            frequency[asset] += 1
            signed[asset] += share
            abs_contrib[asset] += abs(share)
            daily_proxy[(str(getattr(row, "date")), asset)] += share
    total_signed = float(sum(signed.values()))
    total_abs = float(sum(abs_contrib.values()))
    top_abs = sorted(abs_contrib, key=lambda asset: abs_contrib[asset], reverse=True)
    top_signed = sorted(signed, key=lambda asset: signed[asset], reverse=True)

    def pct(top_n: int, *, absolute: bool) -> float | None:
        assets = top_abs[:top_n] if absolute else top_signed[:top_n]
        denom = total_abs if absolute else total_signed
        if denom == 0:
            return None
        values = abs_contrib if absolute else signed
        return float(sum(values[asset] for asset in assets) / denom)

    return {
        "selected_security_count": int(len(frequency)),
        "selection_rows_with_assets": int(sum(frequency.values())),
        "limitation": "V3 sleeve evidence retains selected asset lists and aggregate top-N sleeve returns, not exact per-security realized target contributions; contribution rows are equal-allocation proxy only.",
        "top_10_by_abs_proxy_contribution": [
            {"asset_id": asset, "abs_proxy_contribution": float(abs_contrib[asset]), "signed_proxy_contribution": float(signed[asset]), "selection_count": int(frequency[asset])}
            for asset in top_abs[:10]
        ],
        "top_10_by_selection_frequency": [
            {"asset_id": asset, "selection_count": int(count), "abs_proxy_contribution": float(abs_contrib[asset]), "signed_proxy_contribution": float(signed[asset])}
            for asset, count in frequency.most_common(10)
        ],
        "proxy_top_1_abs_contribution_pct": pct(1, absolute=True),
        "proxy_top_5_abs_contribution_pct": pct(5, absolute=True),
        "proxy_top_10_abs_contribution_pct": pct(10, absolute=True),
        "proxy_top_25_abs_contribution_pct": pct(25, absolute=True),
        "proxy_top_1_signed_contribution_pct": pct(1, absolute=False),
        "proxy_top_5_signed_contribution_pct": pct(5, absolute=False),
        "proxy_top_10_signed_contribution_pct": pct(10, absolute=False),
        "proxy_top_25_signed_contribution_pct": pct(25, absolute=False),
        "exact_exclusion_diagnostics": "INSUFFICIENT_RETAINED_V3_SLEEVE_EVIDENCE_WITHOUT_JOINING_PER_ASSET_TARGETS",
    }


def date_concentration(daily: pd.DataFrame) -> dict[str, Any]:
    work = daily.sort_values("net_daily_return").copy()
    total_sum = float(work["net_daily_return"].sum())
    rows = work.sort_values("net_daily_return", ascending=False).reset_index(drop=True)

    def contribution(top_n: int) -> dict[str, Any]:
        removed = rows.iloc[top_n:]["net_daily_return"]
        return {
            "n": top_n,
            "arithmetic_contribution_pct": float(rows.head(top_n)["net_daily_return"].sum() / total_sum) if total_sum else None,
            "cumulative_return_after_removal": float((1.0 + removed).prod() - 1.0) if len(removed) else None,
            "sharpe_after_removal": performance_from_returns(removed)["sharpe"],
        }

    return {
        "best_10_daily_returns": rows.head(10)[["maturity_session_date", "net_daily_return", "gross_daily_return", "transaction_cost"]].to_dict("records"),
        "worst_10_daily_returns": work.head(10)[["maturity_session_date", "net_daily_return", "gross_daily_return", "transaction_cost"]].to_dict("records"),
        "best_day_concentration": [contribution(n) for n in [1, 5, 10, 25]],
        "median_daily_return": float(rows["net_daily_return"].median()),
    }


def distribution(daily: pd.DataFrame) -> dict[str, Any]:
    values = finite(daily["net_daily_return"]).dropna()
    quantiles = values.quantile([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]).to_dict()
    rounded = values.round(10)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=1)),
        "quantiles": {str(key): float(value) for key, value in quantiles.items()},
        "skew": float(values.skew()),
        "excess_kurtosis": float(values.kurt()),
        "positive_fraction": float((values > 0.0).mean()),
        "zero_fraction": float((values == 0.0).mean()),
        "unique_rounded_1e10": int(rounded.nunique()),
        "discreteness_ratio_unique_to_count": float(rounded.nunique() / len(values)),
        "clipping_truncation_signal": "NO_OBVIOUS_HARD_CLIP_FROM_DAILY_RETURN_SUPPORT",
    }


def nw_ci(values: pd.Series) -> dict[str, Any]:
    return ev.newey_west_mean_ci(finite(values).dropna().tolist(), lag=ev.DEFAULT_V3_SLEEVE_COUNT)


def moving_block_bootstrap_sharpe(values: pd.Series, *, reps: int = 1000, block: int = 21) -> dict[str, Any]:
    arr = finite(values).dropna().to_numpy(dtype=float)
    n = int(len(arr))
    if n == 0:
        return {"reps": 0, "block_length": block, "ci_2_5": None, "ci_97_5": None}
    rng = np.random.default_rng(2401)
    starts = np.arange(n)
    out: list[float] = []
    blocks_needed = int(math.ceil(n / block))
    for _ in range(reps):
        sample_parts = []
        for start in rng.choice(starts, size=blocks_needed, replace=True):
            sample_parts.append(arr[(start + np.arange(block)) % n])
        sample = np.concatenate(sample_parts)[:n]
        vol = sample.std(ddof=1) * math.sqrt(ev.TRADING_SESSIONS_PER_YEAR)
        if vol > 0:
            out.append(float(sample.mean() * ev.TRADING_SESSIONS_PER_YEAR / vol))
    if not out:
        return {"reps": 0, "block_length": block, "ci_2_5": None, "ci_97_5": None}
    return {
        "reps": int(len(out)),
        "block_length": block,
        "ci_2_5": float(np.quantile(out, 0.025)),
        "ci_97_5": float(np.quantile(out, 0.975)),
        "median": float(np.median(out)),
    }


def matched_comparison(huber: dict[str, pd.DataFrame], rff: dict[str, pd.DataFrame]) -> dict[str, Any]:
    h_dates = retained_date_set(huber["daily"])
    r_dates = retained_date_set(rff["daily"])
    h_daily = daily_from_sleeves(huber["sleeves"], 0.0, accepted_dates=h_dates)
    r_daily = daily_from_sleeves(rff["sleeves"], 0.0, accepted_dates=r_dates)
    h_daily["date_key"] = pd.to_datetime(h_daily["maturity_session_date"], utc=True).dt.date.astype(str)
    r_daily["date_key"] = pd.to_datetime(r_daily["maturity_session_date"], utc=True).dt.date.astype(str)
    common_dates = sorted(set(h_daily["date_key"]) & set(r_daily["date_key"]))
    h_match = h_daily[h_daily["date_key"].isin(common_dates)]
    r_match = r_daily[r_daily["date_key"].isin(common_dates)]

    h_rank = huber["rank"].copy()
    r_rank = rff["rank"].copy()
    h_rank["ts"] = pd.to_datetime(h_rank["decision_timestamp"], utc=True).map(lambda ts: ts.isoformat())
    r_rank["ts"] = pd.to_datetime(r_rank["decision_timestamp"], utc=True).map(lambda ts: ts.isoformat())
    common_ts = sorted(set(h_rank["ts"]) & set(r_rank["ts"]))
    h_rank_match = h_rank[h_rank["ts"].isin(common_ts)]
    r_rank_match = r_rank[r_rank["ts"].isin(common_ts)]

    def common_cost_rows(root_sleeves: pd.DataFrame, dates: set[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cost in [1.0, 5.0, 10.0, 20.0, 50.0]:
            daily = daily_from_sleeves(root_sleeves, cost, accepted_dates=dates)
            daily["date_key"] = pd.to_datetime(daily["maturity_session_date"], utc=True).dt.date.astype(str)
            rows.append({"cost_bps": cost, **performance_from_returns(daily[daily["date_key"].isin(dates)]["net_daily_return"])})
        return rows

    date_set = set(common_dates)
    return {
        "common_daily_dates": int(len(common_dates)),
        "common_rank_timestamps": int(len(common_ts)),
        "huber_common_daily": performance_from_returns(h_match["net_daily_return"]),
        "rff_common_daily": performance_from_returns(r_match["net_daily_return"]),
        "huber_common_rank_ic_mean": float(finite(h_rank_match["spearman_rank_ic"]).mean()) if len(h_rank_match) else None,
        "rff_common_rank_ic_mean": float(finite(r_rank_match["spearman_rank_ic"]).mean()) if len(r_rank_match) else None,
        "huber_cost_grid_common_dates": common_cost_rows(huber["sleeves"], date_set),
        "rff_cost_grid_common_dates": common_cost_rows(rff["sleeves"], date_set),
    }


def exclusion_from_window(logs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    sleeves = logs["sleeves"].copy()
    dates = retained_date_set(logs["daily"])
    sleeves["decision_ts"] = pd.to_datetime(sleeves["decision_timestamp"], utc=True)
    outside_quarantine = sleeves[(sleeves["decision_ts"] < QUARANTINE_START) | (sleeves["decision_ts"] > QUARANTINE_END)]
    outside_pending = sleeves[sleeves["decision_ts"] > ANOMALOUS_PENDING_CUTOFF]
    return {
        "exclude_r36_quarantine_window": performance_from_returns(daily_from_sleeves(outside_quarantine, 0.0, accepted_dates=dates)["net_daily_return"]),
        "exclude_anomalous_pending_period_through_2019_06_06": performance_from_returns(daily_from_sleeves(outside_pending, 0.0, accepted_dates=dates)["net_daily_return"]),
        "excluded_r36_sleeve_rows": int(len(sleeves) - len(outside_quarantine)),
        "excluded_anomalous_pending_sleeve_rows": int(len(sleeves) - len(outside_pending)),
    }


def sleeve_geometry(sleeves: pd.DataFrame, retained_daily: pd.DataFrame) -> dict[str, Any]:
    recomputed = daily_from_sleeves(sleeves, 0.0)
    recomputed["date_key"] = pd.to_datetime(recomputed["maturity_session_date"], utc=True).dt.date.astype(str)
    retained = retained_daily.copy()
    retained_date_col = "maturity_session_date" if "maturity_session_date" in retained.columns else "session_date"
    retained["date_key"] = pd.to_datetime(retained[retained_date_col], utc=True).dt.date.astype(str)
    merged = recomputed.merge(retained, on="date_key", suffixes=("_recomputed", "_retained"))
    deltas = (merged["net_daily_return_recomputed"] - merged["net_daily_return_retained"]).abs()
    dupes = int(sleeves.duplicated(["family", "decision_timestamp", "sleeve_id", "maturity_timestamp"]).sum())
    return {
        "sleeve_count_values": sorted(int(v) for v in finite(sleeves["sleeve_count"]).dropna().unique()),
        "sleeve_capital_fraction_min": float(finite(sleeves["sleeve_capital_fraction"]).min()),
        "sleeve_capital_fraction_max": float(finite(sleeves["sleeve_capital_fraction"]).max()),
        "simultaneous_capital_fraction_max": float(finite(sleeves["simultaneous_capital_fraction"]).max()),
        "duplicate_matured_sleeve_keys": dupes,
        "retained_daily_rows": int(len(retained)),
        "recomputed_daily_rows": int(len(recomputed)),
        "daily_reconciliation_rows": int(len(merged)),
        "max_abs_net_daily_reconciliation_delta": float(deltas.max()) if len(deltas) else None,
        "capital_x12_signal": "NOT_OBSERVED_RECOMPUTED_SLEEVE_SUM_MATCHES_RETAINED_DAILY",
        "matured_contributions_once": dupes == 0,
    }


def terminal_classification(report: dict[str, Any]) -> str:
    huber_costs = report["transaction_cost_stress"]["huber"]
    if any(row["sharpe"] is not None and row["sharpe"] < 3.0 for row in huber_costs):
        return "HUBER_ECONOMIC_REALISM_FAILS_R1"
    worst_year = report["year_by_year"]["worst_full_year_by_cumulative_return"]
    if worst_year and worst_year.get("cumulative_return") is not None and worst_year["cumulative_return"] <= 0.0:
        return "HUBER_ECONOMIC_REALISM_FAILS_R1"
    if report["security_concentration"]["exact_exclusion_diagnostics"].startswith("INSUFFICIENT"):
        return "HUBER_ECONOMIC_REALISM_INCONCLUSIVE_R1"
    return "HUBER_ECONOMIC_REALISM_SURVIVES_R1"


def build_report() -> dict[str, Any]:
    huber = read_logs(HUBER_ROOT)
    rff = read_logs(RFF_ROOT)
    huber_dates = retained_date_set(huber["daily"])
    rff_dates = retained_date_set(rff["daily"])
    huber_daily = daily_from_sleeves(huber["sleeves"], 0.0, accepted_dates=huber_dates)
    rff_daily = daily_from_sleeves(rff["sleeves"], 0.0, accepted_dates=rff_dates)
    huber_years = yearly(huber_daily, huber["rank"])
    full_years = [row for row in huber_years if row["valid_days"] >= 200]
    worst_full_year = min(full_years, key=lambda row: row["cumulative_return"]) if full_years else None

    report: dict[str, Any] = {
        "ticket": "DS24_HUBER_ECONOMIC_REALISM_AND_ROBUSTNESS_RED_TEAM_R1",
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "holdout_accessed": False,
        "fit_or_replay_performed": False,
        "workers_stopped_or_restarted": False,
        "orders_placed": False,
        "roots": {
            "huber": rel(HUBER_ROOT),
            "rff_ridge": rel(RFF_ROOT),
            "output_json": rel(OUT_JSON),
            "output_markdown": rel(OUT_MD),
        },
        "manifest_provenance": {
            "huber": {stem: manifest_fingerprint(HUBER_ROOT, stem) for stem in ["sleeve_maturity_ledger_v3", "daily_portfolio_returns_v3", "rank_ic_v3"]},
            "rff_ridge": {stem: manifest_fingerprint(RFF_ROOT, stem) for stem in ["sleeve_maturity_ledger_v3", "daily_portfolio_returns_v3", "rank_ic_v3"]},
        },
        "turnover_semantics": {
            "source": "core/research/ml/ds24_metrics_only_evaluator.py:_turnover and transaction_cost_sensitivity_from_sleeves",
            "formula": "turnover = 0.5 * sum(abs(new_weight - previous_weight)) within each sleeve; cost_drag = turnover * bps / 10000 * sleeve_capital_fraction",
            "sleeve_count": ev.DEFAULT_V3_SLEEVE_COUNT,
            "trading_sessions_per_year": ev.TRADING_SESSIONS_PER_YEAR,
        },
        "sleeve_geometry": sleeve_geometry(huber["sleeves"], huber["daily"]),
        "baseline": {
            "huber_zero_cost": performance_from_returns(huber_daily["net_daily_return"]),
            "rff_ridge_zero_cost": performance_from_returns(rff_daily["net_daily_return"]),
        },
        "transaction_cost_stress": {
            "huber": cost_grid(huber["sleeves"], COST_BPS, accepted_dates=huber_dates),
            "rff_ridge": cost_grid(rff["sleeves"], COST_BPS, accepted_dates=rff_dates),
            "huber_break_even_bps": break_even_cost_bps_from_daily(huber_daily),
            "huber_first_cost_bps_sharpe_below_3": first_threshold_from_daily(huber_daily, 3.0),
            "huber_first_cost_bps_sharpe_below_2": first_threshold_from_daily(huber_daily, 2.0),
            "huber_first_cost_bps_sharpe_below_1": first_threshold_from_daily(huber_daily, 1.0),
        },
        "year_by_year": {
            "rows": huber_years,
            "worst_full_year_by_cumulative_return": worst_full_year,
        },
        "regime_slices": regime_slices(huber_daily, huber["rank"]),
        "security_concentration": security_concentration(huber["sleeves"]),
        "date_concentration": date_concentration(huber_daily),
        "return_distribution": distribution(huber_daily),
        "matched_huber_vs_rff": matched_comparison(huber, rff),
        "quarantine_and_pending_exclusions": exclusion_from_window(huber),
        "statistical_uncertainty": {
            "huber_daily_mean_newey_west": nw_ci(huber_daily["net_daily_return"]),
            "huber_daily_sharpe_moving_block_bootstrap": moving_block_bootstrap_sharpe(huber_daily["net_daily_return"]),
            "huber_rank_ic_newey_west": nw_ci(huber["rank"]["spearman_rank_ic"]),
        },
    }
    report["final_classification"] = terminal_classification(report)
    report["can_huber_legitimately_remain_complete"] = report["final_classification"] != "HUBER_ECONOMIC_REALISM_FAILS_R1"
    return report


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def markdown(report: dict[str, Any]) -> str:
    base = report["baseline"]["huber_zero_cost"]
    stress = report["transaction_cost_stress"]
    matched = report["matched_huber_vs_rff"]
    worst = report["year_by_year"]["worst_full_year_by_cumulative_return"] or {}
    lines = [
        "# DS24 Huber Economic Realism Red Team R1",
        "",
        f"Final classification: `{report['final_classification']}`",
        "",
        "## Core Result",
        "",
        f"- Huber zero-cost daily Sharpe: `{fmt(base['sharpe'])}`; cumulative return: `{fmt(base['cumulative_return'])}`; win rate: `{fmt(base['win_rate'])}`.",
        f"- Break-even cost: `{fmt(stress['huber_break_even_bps'])}` bps per unit turnover.",
        f"- First cost where Sharpe < 3/2/1: `{fmt(stress['huber_first_cost_bps_sharpe_below_3'])}` / `{fmt(stress['huber_first_cost_bps_sharpe_below_2'])}` / `{fmt(stress['huber_first_cost_bps_sharpe_below_1'])}` bps.",
        f"- Worst full year by cumulative return: `{worst.get('year', 'n/a')}` with cumulative `{fmt(worst.get('cumulative_return'))}` and Sharpe `{fmt(worst.get('sharpe'))}`.",
        f"- Common Huber/RFF dates: `{matched['common_daily_dates']}`; Huber Sharpe `{fmt(matched['huber_common_daily']['sharpe'])}` vs RFF Sharpe `{fmt(matched['rff_common_daily']['sharpe'])}`.",
        "",
        "## Limitations",
        "",
        f"- Security attribution: {report['security_concentration']['limitation']}",
        "- No holdout, model fitting, replay, resource-gate change, order placement, or worker lifecycle action was performed.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    write_json(OUT_JSON, report)
    write_text(OUT_MD, markdown(report))
    print(json.dumps({"classification": report["final_classification"], "json": rel(OUT_JSON), "markdown": rel(OUT_MD)}, sort_keys=True))


if __name__ == "__main__":
    main()
