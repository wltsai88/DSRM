# An Agentic Retrieval-Augmented Large Language Model Framework for Trustworthy Governance in Materials Informatics



## Repository Structure

* `data/synthetic\_materials\_2000.json`: Complete benchmark of 2,000 synthetic digital material asset records (1,600 compliant and 400 non-compliant across 4 anomaly families).
* `data/kb\_standards.json`: Curated industrial standards knowledge base passages (ISO/ASTM 52900, ISO/ASTM 52904, GHS/ISO 11014, EAR/ITAR export controls).
* `evaluate\_framework.py`: Exact evaluation script reproducing Table 7 (confusion matrix, Accuracy, Precision, Recall, F1-score, Rules-Only baseline comparison), Figure 3 (EVM gas cost calculation at 68 gas/byte), and Figure 4 (2,880x latency analysis).

## Exact Experimental Parameters (Aligned with Manuscript Table 6)

* **Dataset Size**: 2,000 synthetic records (`1,600` compliant / `400` non-compliant)
* **Embedding Model**: `text-embedding-ada-002` (1,536 dimensions)
* **Vector Retrieval**: FAISS `IndexFlatIP` (cosine similarity), `top-k = 3`
* **Blockchain Simulation**: EVM-compatible permissioned consortium ledger (`Gas\_base = 21,000`, `Gas\_byte = 68 gas/byte`, mean ABI payload = `400 bytes`, `48,200 gas/tx`)

## Quick Start

```bash
python evaluate\_framework.py

