from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def jain_fairness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if np.allclose(x.sum(), 0.0):
        return 0.0
    return float((x.sum() ** 2) / (len(x) * np.square(x).sum()))


def simulate_cost_curve(
    n_transactions: int,
    invalid_ratio: float,
    base_gas: int = 21000,
    storage_gas_per_byte: int = 68,
    avg_payload_bytes: int = 512,
    prevalidation_cost: int = 4500,
) -> Dict[str, float]:
    invalid = int(round(n_transactions * invalid_ratio))
    valid = n_transactions - invalid
    tx_cost = base_gas + storage_gas_per_byte * avg_payload_bytes

    on_chain_only = n_transactions * tx_cost
    proposed = n_transactions * prevalidation_cost + valid * tx_cost
    savings = (on_chain_only - proposed) / max(on_chain_only, 1)

    return {
        "n_transactions": n_transactions,
        "invalid_ratio": invalid_ratio,
        "valid": valid,
        "invalid": invalid,
        "on_chain_only_gas": on_chain_only,
        "proposed_gas": proposed,
        "gas_savings_ratio": savings,
    }


def simulate_negotiation(
    n_pairs: int,
    max_rounds: int,
    concurrency: int,
    seed: int,
    block_time_s: float = 2.0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    buyer_res = rng.uniform(0.55, 1.00, size=n_pairs)
    seller_res = rng.uniform(0.10, 0.70, size=n_pairs)
    buyer_pat = rng.uniform(0.80, 0.98, size=n_pairs)
    seller_pat = rng.uniform(0.80, 0.98, size=n_pairs)
    init_price = rng.uniform(0.40, 0.90, size=n_pairs)

    deal_rounds: List[int] = []
    deal_prices: List[float] = []
    buyer_utils: List[float] = []
    seller_utils: List[float] = []

    for i in range(n_pairs):
        price = init_price[i]
        deal_found = False
        for t in range(1, max_rounds + 1):
            buyer_offer = buyer_res[i] * (buyer_pat[i] ** (t - 1))
            seller_offer = seller_res[i] + (price - seller_res[i]) * (seller_pat[i] ** (t - 1))
            if buyer_offer >= seller_offer:
                agreed = 0.5 * (buyer_offer + seller_offer)
                deal_rounds.append(t)
                deal_prices.append(agreed)
                buyer_utils.append(max(0.0, buyer_res[i] - agreed))
                seller_utils.append(max(0.0, agreed - seller_res[i]))
                deal_found = True
                break
        if not deal_found:
            deal_rounds.append(max_rounds + 1)
            deal_prices.append(np.nan)
            buyer_utils.append(0.0)
            seller_utils.append(0.0)

    agreements = np.array([r <= max_rounds for r in deal_rounds], dtype=int)
    rounds = np.array(deal_rounds, dtype=float)

    # crude queuing penalty for concurrency
    queue_penalty = np.log1p(concurrency) * 0.8
    latency = np.where(agreements == 1, rounds * 1.25 + block_time_s + queue_penalty, np.nan)

    return {
        "n_pairs": n_pairs,
        "concurrency": concurrency,
        "agreement_rate": float(np.mean(agreements)),
        "avg_rounds_to_deal": float(np.nanmean(np.where(agreements == 1, rounds, np.nan))),
        "avg_latency_s": float(np.nanmean(latency)),
        "buyer_utility_mean": float(np.mean(buyer_utils)),
        "seller_utility_mean": float(np.mean(seller_utils)),
        "social_welfare_mean": float(np.mean(np.array(buyer_utils) + np.array(seller_utils))),
        "fairness_jain": float(jain_fairness(np.array(buyer_utils) + 1e-9) + jain_fairness(np.array(seller_utils) + 1e-9)) / 2.0,
    }


def plot_curve(df: pd.DataFrame, x: str, y: str, out_path: str, xlabel: str, ylabel: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(df[x], df[y], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_errorbar(
    df: pd.DataFrame,
    x: str,
    y: str,
    yerr: str,
    out_path: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.errorbar(df[x], df[y], yerr=df[yerr], marker="o", capsize=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # 1) Cost sensitivity to invalid ratio
    cost_rows = []
    for invalid_ratio in [0.05, 0.10, 0.20, 0.30, 0.40]:
        for n in [500, 1000, 2000, 5000, 10000]:
            cost_rows.append(simulate_cost_curve(n_transactions=n, invalid_ratio=invalid_ratio))
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(Path(args.out_dir) / "cost_sensitivity.csv", index=False)

    cost_2k = cost_df[cost_df["n_transactions"] == 2000].copy().sort_values("invalid_ratio")
    plt.figure(figsize=(8, 5))
    plt.plot(cost_2k["invalid_ratio"], cost_2k["gas_savings_ratio"] * 100, marker="o")
    plt.axhline(0, linestyle="--")
    plt.axvline(0.0806, linestyle="--")
    plt.xlabel("Invalid submission ratio")
    plt.ylabel("Gas savings (%)")
    plt.title("Cost benefit of off-chain pre-validation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(args.out_dir) / "fig3_gas_savings_vs_invalid_ratio.png", dpi=300)
    plt.close()

    # 2) Negotiation / latency sensitivity to concurrency
    neg_rows: List[Dict[str, float]] = []
    for concurrency in [10, 50, 100, 500, 1000, 2000]:
        for rep in range(10):
            row = simulate_negotiation(
                n_pairs=1000,
                max_rounds=20,
                concurrency=concurrency,
                seed=args.seed + rep + concurrency,
            )
            row["rep"] = rep
            neg_rows.append(row)
    neg_df = pd.DataFrame(neg_rows)
    neg_df.to_csv(Path(args.out_dir) / "negotiation_latency.csv", index=False)

    neg_summary = (
        neg_df.groupby("concurrency", as_index=False)
        .agg(
            agreement_rate_mean=("agreement_rate", "mean"),
            agreement_rate_std=("agreement_rate", "std"),
            avg_latency_s_mean=("avg_latency_s", "mean"),
            avg_latency_s_std=("avg_latency_s", "std"),
            social_welfare_mean=("social_welfare_mean", "mean"),
            fairness_jain_mean=("fairness_jain", "mean"),
        )
    )
    neg_summary.to_csv(Path(args.out_dir) / "negotiation_summary.csv", index=False)

    plot_errorbar(
        neg_summary,
        x="concurrency",
        y="avg_latency_s_mean",
        yerr="avg_latency_s_std",
        out_path=str(Path(args.out_dir) / "fig4_latency_vs_concurrency.png"),
        xlabel="Concurrent requests",
        ylabel="Mean latency (s)",
        title="Negotiation latency under increasing concurrency",
    )

    plot_errorbar(
        neg_summary,
        x="concurrency",
        y="agreement_rate_mean",
        yerr="agreement_rate_std",
        out_path=str(Path(args.out_dir) / "fig5_agreement_vs_concurrency.png"),
        xlabel="Concurrent requests",
        ylabel="Agreement rate",
        title="Agreement rate under increasing concurrency",
    )

    summary = {
        "cost_rows": len(cost_df),
        "negotiation_rows": len(neg_df),
        "files": [
            "cost_sensitivity.csv",
            "fig3_gas_savings_vs_invalid_ratio.png",
            "negotiation_latency.csv",
            "negotiation_summary.csv",
            "fig4_latency_vs_concurrency.png",
            "fig5_agreement_vs_concurrency.png",
        ],
    }
    with open(Path(args.out_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved outputs to", args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
