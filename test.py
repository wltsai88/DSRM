import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Figure 3: Gas savings vs invalid submission ratio
# ---------------------------
invalid_ratio = np.array([0.05, 0.10, 0.20, 0.30, 0.40])
gas_savings = np.array([-3.0, 2.0, 12.0, 22.0, 32.0])

plt.figure(figsize=(10, 6))
plt.plot(invalid_ratio, gas_savings, marker='o', linewidth=2.0, markersize=8)
plt.axhline(0, linestyle='--', linewidth=1.5)
plt.axvline(0.08, linestyle='--', linewidth=1.5)
plt.title("Cost benefit of off-chain pre-validation", fontsize=20, pad=16)
plt.xlabel("Invalid submission ratio", fontsize=15)
plt.ylabel("Gas savings (%)", fontsize=15)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig3_gas_savings_vs_invalid_ratio.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------
# Figure 4: Negotiation latency under increasing concurrency
# ---------------------------
concurrency_latency = np.array([10, 50, 100, 500, 1000, 2000])
mean_latency = np.array([5.35, 6.52, 7.08, 8.34, 8.92, 9.45])
latency_err = np.array([0.05, 0.06, 0.05, 0.04, 0.04, 0.03])

plt.figure(figsize=(10, 6))
plt.errorbar(
    concurrency_latency,
    mean_latency,
    yerr=latency_err,
    fmt='o-',
    linewidth=2.0,
    markersize=8,
    capsize=5
)
plt.title("Negotiation latency under increasing concurrency", fontsize=20, pad=16)
plt.xlabel("Concurrent requests", fontsize=15)
plt.ylabel("Mean latency (s)", fontsize=15)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig4_latency_vs_concurrency.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------
# Figure 5: Agreement rate under increasing concurrency
# ---------------------------
concurrency_agreement = np.array([10, 50, 100, 500, 1000, 2000])
agreement_rate = np.array([0.751, 0.754, 0.746, 0.755, 0.750, 0.756])
agreement_err = np.array([0.0195, 0.0130, 0.0170, 0.0130, 0.0180, 0.0153])

plt.figure(figsize=(10, 6))
plt.errorbar(
    concurrency_agreement,
    agreement_rate,
    yerr=agreement_err,
    fmt='o-',
    linewidth=2.0,
    markersize=8,
    capsize=6
)
plt.title("Agreement rate under increasing concurrency", fontsize=20, pad=16)
plt.xlabel("Concurrent requests", fontsize=15)
plt.ylabel("Agreement rate", fontsize=15)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig5_agreement_vs_concurrency.png", dpi=300, bbox_inches="tight")
plt.show()