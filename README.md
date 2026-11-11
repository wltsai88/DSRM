# Experiment starter kit

## Files

- `run_compliance_benchmark.py`
  - Evaluates rule-based, TF-IDF, BM25, dense retrieval, hybrid retrieval, and optional LLM-backed RAG compliance classification.
  - Also computes retrieval metrics when gold evidence doc IDs are available.

- `simulate_negotiation_and_chain.py`
  - Simulates on-chain/off-chain cost, latency, agreement rate, social welfare, and fairness under different invalid ratios and concurrent loads.
  - Useful for turning the current single-point results into parameter sweeps and sensitivity analyses.

- `requirements.txt`
  - Python dependencies.

## Suggested dataset format

### 1) Policy / standards corpus

Create a directory `policy_docs/` with one `.txt` or `.md` file per document. Example:

```text
policy_docs/
  ISO_ASTM_52900.txt
  export_control_rule_A.txt
  SDS_minimum_requirements.txt
  license_template_quota.txt
```

### 2) Asset benchmark

Create a JSONL file `assets.jsonl`.
Each line should look like:

```json
{
  "asset_id": "A0001",
  "text": "Ti-6Al-4V powder bed fusion design. Includes composition, laser power, scan speed, heat treatment, and license cap of 500 units.",
  "label": 1,
  "material_family": "titanium_alloy",
  "evidence_doc_ids": ["SDS_minimum_requirements", "license_template_quota"],
  "meta": {
    "origin": "MaterialsProject/JARVIS/manual",
    "jurisdiction": "US",
    "version": "2026-01"
  }
}
```

Label meaning:
- `1` = compliant
- `0` = non-compliant

## Recommended experiments for the revised paper

### A. Compliance oracle benchmark

Run at least these baselines:

1. Keyword/rule baseline
2. TF-IDF + Logistic Regression
3. BM25-only retrieval + heuristic decision
4. Dense retrieval + heuristic decision
5. Hybrid retrieval + heuristic decision
6. LLM-only (no retrieval)
7. Vanilla RAG
8. Proposed agentic RAG with policy checks / HITL threshold

Suggested command:

```bash
python run_compliance_benchmark.py \
  --assets assets.jsonl \
  --policy_dir policy_docs \
  --out_dir results_compliance \
  --dense_model BAAI/bge-small-en-v1.5 \
  --top_k 5
```

Optional OpenAI / compatible endpoint:

```bash
export OPENAI_API_KEY=...
python run_compliance_benchmark.py \
  --assets assets.jsonl \
  --policy_dir policy_docs \
  --out_dir results_compliance_openai \
  --llm_backend openai \
  --llm_model gpt-4.1-mini \
  --top_k 5
```

### B. Sensitivity / ablation study

Sweep these:
- `top_k`: 1, 3, 5, 10
- chunk size: 256, 512, 1024 tokens
- invalid ratio: 5%, 10%, 20%, 30%, 40%
- KB size: 500, 1000, 2500, 5000+
- confidence threshold for HITL escalation

### C. Negotiation + blockchain study

```bash
python simulate_negotiation_and_chain.py --out_dir results_chain
```

This produces:
- gas cost vs invalid ratio
- latency vs concurrency
- agreement rate
- social welfare
- Jain fairness index

## What to report in the paper

For `run_compliance_benchmark.py`, report:
- Accuracy, Precision, Recall, F1, AUROC
- Recall@k, MRR, nDCG (if evidence labels exist)
- mean latency and cost per query
- category-wise errors by anomaly type

For `simulate_negotiation_and_chain.py`, report:
- mean / std across multiple random seeds
- sensitivity curves instead of a single best number
- comparison against at least three baselines:
  - on-chain-only
  - centralized pre-check + chain logging
  - proposed off-chain intelligent pre-validation

## Paper-writing advice

A strong resubmission should state clearly:
- exact corpus sources
- exact model names / versions
- prompt templates
- random seeds
- hardware
- code / data availability statement

