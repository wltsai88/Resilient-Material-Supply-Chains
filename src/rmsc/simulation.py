"""Separate analytical models. No measured gas, blockchain or network latency."""
import numpy as np

from .evaluation import write_csv


def cost_scenarios(n=3255):
    tx, pre = 21000 + 512 * 68, 4500
    result = []
    for q in (.05, .10, .20, .30, .40):
        invalid = round(n * q)  # Python round, ties to even
        baseline = n * tx
        checked = n * pre + (n - invalid) * tx
        result.append({"requests": n, "invalid_ratio": q, "invalid_requests": invalid,
                       "cost_per_transaction": tx, "precheck_per_request": pre,
                       "baseline_cost": baseline, "prevalidated_cost": checked,
                       "reduction_fraction": (baseline - checked) / baseline,
                       "continuous_break_even_ratio": pre / tx})
    return result


def negotiate(b, s, p0, buyer_decay, seller_decay, max_rounds=20):
    for t in range(1, max_rounds + 1):
        buyer = b * buyer_decay ** (t - 1)
        seller = s + (p0 - s) * seller_decay ** (t - 1)
        if buyer >= seller:
            return t, (buyer + seller) / 2
    return None, None


def bargaining(out, n=3255, repetitions=10):
    pair_rows, run_rows, summaries = [], [], []
    for concurrency in (10, 50, 100, 500, 1000, 2000):
        means, rates = [], []
        for repetition in range(repetitions):
            seed = 42 + repetition + concurrency
            rng = np.random.default_rng(seed)
            # Documented new draw order: one vector per parameter, then pairs.
            b = rng.uniform(.55, 1., n)
            s = rng.uniform(.10, .70, n)
            p0 = rng.uniform(.40, .90, n)
            db, ds = rng.uniform(.80, .98, (2, n))
            success_latencies = []
            for i in range(n):
                t, price = negotiate(b[i], s[i], p0[i], db[i], ds[i])
                latency = 1.25*t + 2. + .8*np.log1p(concurrency) if t is not None else None
                if latency is not None:
                    success_latencies.append(float(latency))
                pair_rows.append({"concurrency": concurrency, "repetition": repetition,
                                  "seed": seed, "pair_id": i+1, "buyer": b[i], "seller": s[i],
                                  "initial_price": p0[i], "buyer_decay": db[i], "seller_decay": ds[i],
                                  "agreed": int(t is not None), "agreement_round": t,
                                  "agreement_price": price, "modeled_latency_seconds": latency})
            mean = float(np.mean(success_latencies)) if success_latencies else None
            rate = len(success_latencies) / n
            means.append(mean)
            rates.append(rate)
            run_rows.append({"concurrency": concurrency, "repetition": repetition, "seed": seed,
                             "successful_pairs": len(success_latencies), "pairs": n,
                             "agreement_rate": rate, "conditional_latency_mean": mean})
        summaries.append({"concurrency": concurrency, "repetitions": repetitions,
                          "conditional_latency_mean": float(np.mean(means)),
                          "latency_run_mean_sd": float(np.std(means, ddof=1)),
                          "agreement_rate_mean": float(np.mean(rates)),
                          "agreement_rate_run_sd": float(np.std(rates, ddof=1))})
    write_csv(out / "bargaining_pairs.csv", pair_rows)
    write_csv(out / "bargaining_runs.csv", run_rows)
    write_csv(out / "bargaining_summary.csv", summaries)
    return summaries
