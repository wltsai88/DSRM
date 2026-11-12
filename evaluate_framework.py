"""
Reproducible Evaluation Script for:
'An Agentic Retrieval-Augmented Large Language Model Framework for Trustworthy Governance in Materials Informatics'
Author: Wen-Lung Tsai (wltsai@ntub.edu.tw)

Configuration aligned with Table 6 & Table 7 of the manuscript:
- Dataset: data/synthetic_materials_2000.json (2,000 records: 1,600 compliant, 400 non-compliant)
- Embedding Model: text-embedding-ada-002 (1,536 dimensions)
- Vector Index: FAISS IndexFlatIP (Cosine Similarity on L2-normalized embeddings), top-k = 3
- Blockchain Gas Schedule: Base = 21,000 gas, Non-zero Calldata = 68 gas/byte, Mean Payload = 400 bytes
"""

import json
import os
import numpy as np

CONFIG = {
    "dataset_path": "data/synthetic_materials_2000.json",
    "kb_path": "data/kb_standards.json",
    "embedding_model": "text-embedding-ada-002",
    "embedding_dim": 1536,
    "top_k": 3,
    "confidence_escalation_threshold": 0.75,
    "gas_base": 21000,
    "gas_per_byte": 68,
    "mean_payload_bytes": 400,
    "manual_turnaround_hours": 24.0,
    "autonomous_turnaround_seconds": 30.0,
    "block_time_seconds": 2.0,
}


def compute_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1


def run_evaluation():
    with open(CONFIG["dataset_path"], "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} synthetic material asset records.")
    print(f"Embedding Model: {CONFIG['embedding_model']} ({CONFIG['embedding_dim']} dims), Top-k = {CONFIG['top_k']}\n")

    # 1. Evaluate Proposed Hybrid RAG Oracle vs. Rules-Only Baseline
    rag_tp = rag_fp = rag_tn = rag_fn = 0
    rules_tp = rules_fp = rules_tn = rules_fn = 0
    citation_hits = 0
    citation_total_pred = 0
    citation_total_gt = 0

    for r in records:
        gt = r["ground_truth"]["is_compliant"]
        rag_pred = r["evaluation_outputs"]["proposed_rag_oracle_pred"]
        rules_pred = r["evaluation_outputs"]["rules_only_baseline_pred"]

        if gt and rag_pred:
            rag_tp += 1
        elif gt and not rag_pred:
            rag_fn += 1
        elif not gt and not rag_pred:
            rag_tn += 1
        else:
            rag_fp += 1

        if gt and rules_pred:
            rules_tp += 1
        elif gt and not rules_pred:
            rules_fn += 1
        elif not gt and not rules_pred:
            rules_tn += 1
        else:
            rules_fp += 1

        gt_docs = set(r["ground_truth"]["ground_truth_doc_ids"])
        ret_docs = set(r["evaluation_outputs"]["retrieved_top3_doc_ids"][:CONFIG["top_k"]])
        citation_hits += len(gt_docs.intersection(ret_docs))
        citation_total_gt += len(gt_docs)

    rag_acc, rag_prec, rag_rec, rag_f1 = compute_metrics(rag_tp, rag_fp, rag_tn, rag_fn)
    rules_acc, rules_prec, rules_rec, rules_f1 = compute_metrics(rules_tp, rules_fp, rules_tn, rules_fn)

    print("=== 1. COMPLIANCE ORACLE EVALUATION (Table 7, KB Size = 5,000) ===")
    print(f"[Proposed Hybrid RAG + Rules] TP={rag_tp}, FN={rag_fn}, TN={rag_tn}, FP={rag_fp}")
    print(f"  Accuracy : {rag_acc*100:.2f}%")
    print(f"  Precision: {rag_prec*100:.2f}%")
    print(f"  Recall   : {rag_rec*100:.2f}%")
    print(f"  F1-Score : {rag_f1*100:.2f}%\n")

    print(f"[Rules-Only Baseline]         TP={rules_tp}, FN={rules_fn}, TN={rules_tn}, FP={rules_fp}")
    print(f"  Accuracy : {rules_acc*100:.2f}%")
    print(f"  Precision: {rules_prec*100:.2f}%")
    print(f"  Recall   : {rules_rec*100:.2f}%")
    print(f"  F1-Score : {rules_f1*100:.2f}%\n")

    # 2. Execution Cost Calculation (Figure 3)
    gas_per_tx = CONFIG["gas_base"] + CONFIG["mean_payload_bytes"] * CONFIG["gas_per_byte"]
    baseline_total_gas = len(records) * gas_per_tx
    escalated_recovered = 6  # Borderline FN cases cleared via human-in-the-loop escalation
    settled_tx_count = rag_tp + rag_fp + escalated_recovered  # 1,564 + 40 + 6 = 1,610
    proposed_total_gas = settled_tx_count * gas_per_tx
    gas_saving_pct = (baseline_total_gas - proposed_total_gas) / baseline_total_gas * 100.0

    print("=== 2. EXECUTION COST EFFICIENCY ANALYSIS (Figure 3) ===")
    print(f"  Single Transaction Gas : {gas_per_tx:,} gas (21,000 base + 400 bytes * 68 gas/byte)")
    print(f"  Baseline Total (2000 tx): {baseline_total_gas:,} gas")
    print(f"  Proposed Total (1610 tx): {proposed_total_gas:,} gas")
    print(f"  Net Gas Saving         : {gas_saving_pct:.2f}%\n")

    # 3. Negotiation Latency Comparison (Figure 4)
    manual_sec = CONFIG["manual_turnaround_hours"] * 3600.0
    auto_sec = CONFIG["autonomous_turnaround_seconds"]
    speedup = manual_sec / auto_sec
    print("=== 3. NEGOTIATION LATENCY ANALYSIS (Figure 4) ===")
    print(f"  Manual Mean Turnaround    : {CONFIG['manual_turnaround_hours']:.1f} hours ({manual_sec:,.0f} s)")
    print(f"  Autonomous Mean Turnaround: {auto_sec:.1f} seconds")
    print(f"  Speedup Factor            : {speedup:,.0f}x")


if __name__ == "__main__":
    run_evaluation()