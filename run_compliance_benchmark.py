from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    ndcg_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import minmax_scale


TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\.]+")
DEFAULT_RESTRICTED = {
    "export controlled",
    "missing sds",
    "restricted alloy",
    "duplicate hash",
    "sanctioned destination",
    "no safety data sheet",
}


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_default_env_files() -> None:
    for env_path in [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]:
        load_env_file(env_path)


@dataclass
class Asset:
    asset_id: str
    text: str
    label: int
    material_family: str
    evidence_doc_ids: List[str]
    meta: Dict[str, object]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def load_assets(path: str) -> List[Asset]:
    assets: List[Asset] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assets.append(
                Asset(
                    asset_id=str(row["asset_id"]),
                    text=str(row["text"]),
                    label=int(row["label"]),
                    material_family=str(row.get("material_family", "unknown")),
                    evidence_doc_ids=[str(x) for x in row.get("evidence_doc_ids", [])],
                    meta=dict(row.get("meta", {})),
                )
            )
    if not assets:
        raise ValueError(f"No assets loaded from {path}")
    return assets


def resolve_n_splits(assets: Sequence[Asset], requested_n_splits: int) -> int:
    if requested_n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    labels = [a.label for a in assets]
    _, counts = np.unique(labels, return_counts=True)
    max_splits = int(counts.min())
    if max_splits < 2:
        raise ValueError("Need at least 2 samples in each class for StratifiedKFold")
    return min(requested_n_splits, max_splits)


def chunk_text(text: str, max_tokens: int = 256, overlap: int = 32) -> List[str]:
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    chunks: List[str] = []
    start = 0
    step = max(1, max_tokens - overlap)
    while start < len(words):
        end = min(len(words), start + max_tokens)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


def load_policy_chunks(policy_dir: str, chunk_size: int = 256, overlap: int = 32) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in sorted(Path(policy_dir).glob("*")):
        if not path.is_file():
            continue
        doc_id = path.stem
        text = path.read_text(encoding="utf-8", errors="ignore")
        for idx, chunk in enumerate(chunk_text(text, max_tokens=chunk_size, overlap=overlap)):
            chunks.append(Chunk(chunk_id=f"{doc_id}::c{idx}", doc_id=doc_id, text=chunk))
    if not chunks:
        raise ValueError(f"No policy chunks found in {policy_dir}")
    return chunks


class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        self.corpus_tokens = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order]


class DenseRetriever:
    def __init__(self, chunks: Sequence[Chunk], model_name: str):
        from sentence_transformers import SentenceTransformer

        self.chunks = list(chunks)
        self.model = SentenceTransformer(model_name)
        self.emb = self.model.encode(
            [c.text for c in self.chunks],
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = np.dot(self.emb, q)
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order]


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, alpha: float = 0.5):
        self.bm25 = bm25
        self.dense = dense
        self.alpha = alpha
        self.chunks = bm25.chunks

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        bm25_scores = self.bm25.bm25.get_scores(tokenize(query))
        q = self.dense.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        dense_scores = np.dot(self.dense.emb, q)
        b = minmax_scale(bm25_scores)
        d = minmax_scale(dense_scores)
        combo = self.alpha * b + (1.0 - self.alpha) * d
        order = np.argsort(combo)[::-1][:top_k]
        return [(self.chunks[i], float(combo[i])) for i in order]


class RuleBaseline:
    def __init__(self, restricted_terms: Optional[Iterable[str]] = None):
        self.restricted_terms = {t.lower() for t in (restricted_terms or DEFAULT_RESTRICTED)}

    def predict(self, text: str) -> Tuple[int, float, List[str]]:
        text_low = text.lower()
        hits = [term for term in self.restricted_terms if term in text_low]
        score = 0.95 if hits else 0.05
        # 1 means compliant; 0 means non-compliant
        return (0 if hits else 1, score, hits)


class TfidfClassifier:
    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.clf = LogisticRegression(max_iter=300, class_weight="balanced")

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        x = self.vec.fit_transform(texts)
        self.clf.fit(x, labels)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        x = self.vec.transform(texts)
        return self.clf.predict_proba(x)[:, 1]

    def predict(self, texts: Sequence[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(texts) >= threshold).astype(int)


class RetrievalAugmentedTfidfClassifier:
    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.clf = LogisticRegression(max_iter=500, class_weight="balanced")

    @staticmethod
    def build_text(asset_text: str, retrieved: Sequence[Tuple[Chunk, float]]) -> str:
        evidence = "\n".join(f"[{c.doc_id}] {c.text}" for c, _ in retrieved)
        return f"{asset_text}\n\nPOLICY_EVIDENCE:\n{evidence}"

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        x = self.vec.fit_transform(texts)
        self.clf.fit(x, labels)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        x = self.vec.transform(texts)
        return self.clf.predict_proba(x)[:, 1]


class OptionalOpenAIRAG:
    def __init__(self, model: str, backend: str):
        self.model = model
        self.backend = backend
        self.enabled = False
        self.client = None
        try:
            if backend == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    from openai import OpenAI

                    self.client = OpenAI(api_key=api_key)
                    self.enabled = True
            elif backend == "azure_openai":
                api_key = os.getenv("AZURE_OPENAI_API_KEY")
                endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
                if api_key and endpoint:
                    from openai import AzureOpenAI

                    self.client = AzureOpenAI(
                        api_key=api_key,
                        azure_endpoint=endpoint,
                        api_version=api_version,
                    )
                    self.enabled = True
        except Exception:
            self.enabled = False

    def predict(self, asset_text: str, evidence: Sequence[Chunk]) -> Tuple[int, float, str]:
        if not self.enabled:
            raise RuntimeError("LLM backend is not enabled")

        context = "\n\n".join([f"[{c.doc_id}] {c.text}" for c in evidence])
        prompt = f"""
You are a policy compliance oracle for materials informatics and distributed manufacturing.
Determine whether the asset is COMPLIANT or NON_COMPLIANT based only on the evidence.
Return JSON with keys: label, confidence, rationale.
Label must be either COMPLIANT or NON_COMPLIANT.

ASSET:
{asset_text}

EVIDENCE:
{context}
""".strip()
        try:
            resp = self.client.responses.create(model=self.model, input=prompt)
            text = resp.output_text.strip()
            low = text.lower()
            label = 1 if "compliant" in low and "non_compliant" not in low else 0
            conf_match = re.search(r'"?confidence"?\s*[:=]\s*([0-9]*\.?[0-9]+)', text)
            conf = float(conf_match.group(1)) if conf_match else 0.5
            return label, conf, text
        except Exception as exc:
            raise RuntimeError(f"openai-error: {exc}") from exc


def retrieval_metrics(
    assets: Sequence[Asset],
    retriever,
    top_k: int,
) -> Dict[str, float]:
    recall_hits = []
    rr_values = []
    ndcg_values = []
    evaluated = 0
    for asset in assets:
        if not asset.evidence_doc_ids:
            continue
        gold = set(asset.evidence_doc_ids)
        results = retriever.search(asset.text, top_k=top_k)
        ranked_docs = [c.doc_id for c, _ in results]
        hits = [1 if doc_id in gold else 0 for doc_id in ranked_docs]
        recall_hits.append(int(any(hits)))
        rr = 0.0
        for rank, h in enumerate(hits, start=1):
            if h:
                rr = 1.0 / rank
                break
        rr_values.append(rr)
        ideal = sorted(hits, reverse=True)
        if sum(ideal) == 0:
            ndcg_values.append(0.0)
        else:
            ndcg_values.append(float(ndcg_score([ideal], [hits])))
        evaluated += 1
    if evaluated == 0:
        return {"retrieval_evaluated": 0}
    return {
        "retrieval_evaluated": evaluated,
        f"Recall@{top_k}": float(np.mean(recall_hits)),
        "MRR": float(np.mean(rr_values)),
        "nDCG": float(np.mean(ndcg_values)),
    }


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_prob: Sequence[float]) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["auprc"] = float("nan")
    return out


def evaluate_rag_cv(
    assets: Sequence[Asset],
    retriever,
    top_k: int = 5,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    y = np.array([a.label for a in assets])
    idx = np.arange(len(assets))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_rows = []
    pred_rows = []

    for fold, (tr, te) in enumerate(skf.split(idx, y), start=1):
        model = RetrievalAugmentedTfidfClassifier()

        train_texts = []
        for i in tr:
            asset = assets[i]
            retrieved = retriever.search(asset.text, top_k=top_k)
            train_texts.append(model.build_text(asset.text, retrieved))

        model.fit(train_texts, y[tr])

        test_texts = []
        cached_retrieval = []
        for i in te:
            asset = assets[i]
            retrieved = retriever.search(asset.text, top_k=top_k)
            cached_retrieval.append(retrieved)
            test_texts.append(model.build_text(asset.text, retrieved))

        probs = model.predict_proba(test_texts)
        preds = (probs >= 0.5).astype(int)

        row = classification_metrics(y[te], preds, probs)
        row["fold"] = fold
        fold_rows.append(row)

        for local_j, i in enumerate(te):
            asset = assets[i]
            pred_rows.append(
                {
                    "system": "rag_tfidf_cv",
                    "fold": fold,
                    "asset_id": asset.asset_id,
                    "y_true": int(asset.label),
                    "y_pred": int(preds[local_j]),
                    "score": float(probs[local_j]),
                    "details": " | ".join(
                        f"{c.doc_id}:{score:.4f}" for c, score in cached_retrieval[local_j]
                    ),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    summary: Dict[str, float] = {}
    for col in ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]:
        summary[col] = float(fold_df[col].mean())
        summary[f"{col}_std"] = float(fold_df[col].std())

    return summary, pd.DataFrame(pred_rows)


def summarize_fold_metrics(fold_rows: Sequence[Dict[str, float]], metric_cols: Sequence[str]) -> Dict[str, float]:
    fold_df = pd.DataFrame(fold_rows)
    summary: Dict[str, float] = {}
    for col in metric_cols:
        summary[col] = float(fold_df[col].mean())
        summary[f"{col}_std"] = float(fold_df[col].std())
    return summary


def evaluate_rules_cv(assets: Sequence[Asset], rules: RuleBaseline, splitter) -> Tuple[Dict[str, float], pd.DataFrame]:
    y = np.array([a.label for a in assets])
    idx = np.arange(len(assets))
    fold_rows = []
    pred_rows = []

    for fold, (_, te) in enumerate(splitter.split(idx, y), start=1):
        y_true = y[te]
        y_pred = []
        y_prob = []
        for i in te:
            asset = assets[i]
            pred, score, hits = rules.predict(asset.text)
            prob = score if pred == 1 else 1.0 - score
            y_pred.append(pred)
            y_prob.append(prob)
            pred_rows.append(
                {
                    "system": "rules_cv",
                    "fold": fold,
                    "asset_id": asset.asset_id,
                    "y_true": int(asset.label),
                    "y_pred": int(pred),
                    "score": float(prob),
                    "details": ", ".join(hits),
                }
            )
        row = classification_metrics(y_true, y_pred, y_prob)
        row["fold"] = fold
        fold_rows.append(row)

    return summarize_fold_metrics(fold_rows, ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]), pd.DataFrame(pred_rows)


def evaluate_tfidf_cv(assets: Sequence[Asset], splitter) -> Tuple[Dict[str, float], pd.DataFrame]:
    y = np.array([a.label for a in assets])
    idx = np.arange(len(assets))
    fold_rows = []
    pred_rows = []

    for fold, (tr, te) in enumerate(splitter.split(idx, y), start=1):
        model = TfidfClassifier()
        model.fit([assets[i].text for i in tr], y[tr])
        probs = model.predict_proba([assets[i].text for i in te])
        preds = (probs >= 0.5).astype(int)
        row = classification_metrics(y[te], preds, probs)
        row["fold"] = fold
        fold_rows.append(row)

        for local_j, i in enumerate(te):
            asset = assets[i]
            pred_rows.append(
                {
                    "system": "tfidf_cv",
                    "fold": fold,
                    "asset_id": asset.asset_id,
                    "y_true": int(asset.label),
                    "y_pred": int(preds[local_j]),
                    "score": float(probs[local_j]),
                    "details": "",
                }
            )

    return summarize_fold_metrics(fold_rows, ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]), pd.DataFrame(pred_rows)


def evaluate_retrieval_cv(
    assets: Sequence[Asset],
    retriever,
    top_k: int,
    splitter,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    y = np.array([a.label for a in assets])
    idx = np.arange(len(assets))
    fold_rows = []
    pred_rows = []

    for fold, (_, te) in enumerate(splitter.split(idx, y), start=1):
        fold_assets = [assets[i] for i in te]
        row = retrieval_metrics(fold_assets, retriever, top_k)
        row["fold"] = fold
        fold_rows.append(row)

        for asset in fold_assets:
            retrieved = retriever.search(asset.text, top_k=top_k)
            pred_rows.append(
                {
                    "system": "retrieval_cv",
                    "fold": fold,
                    "asset_id": asset.asset_id,
                    "y_true": int(asset.label),
                    "y_pred": "",
                    "score": "",
                    "details": " | ".join(f"{c.doc_id}:{score:.4f}" for c, score in retrieved),
                }
            )

    metric_cols = [f"Recall@{top_k}", "MRR", "nDCG", "retrieval_evaluated"]
    return summarize_fold_metrics(fold_rows, metric_cols), pd.DataFrame(pred_rows)


def evaluate_llm_rag_cv(
    assets: Sequence[Asset],
    retriever,
    llm_rag: OptionalOpenAIRAG,
    top_k: int,
    splitter,
) -> Tuple[Optional[Dict[str, float]], pd.DataFrame, Optional[str]]:
    y = np.array([a.label for a in assets])
    idx = np.arange(len(assets))
    fold_rows = []
    pred_rows = []

    for fold, (_, te) in enumerate(splitter.split(idx, y), start=1):
        preds = []
        probs = []
        for i in te:
            asset = assets[i]
            retrieved = retriever.search(asset.text, top_k=top_k)
            try:
                pred, conf, raw = llm_rag.predict(asset.text, [c for c, _ in retrieved])
            except Exception as exc:
                return None, pd.DataFrame(pred_rows), str(exc)
            prob = conf if pred == 1 else 1.0 - conf
            preds.append(pred)
            probs.append(prob)
            pred_rows.append(
                {
                    "system": "llm_hybrid_rag_cv",
                    "fold": fold,
                    "asset_id": asset.asset_id,
                    "y_true": int(asset.label),
                    "y_pred": int(pred),
                    "score": float(prob),
                    "details": raw[:1000],
                }
            )
        row = classification_metrics(y[te], preds, probs)
        row["fold"] = fold
        fold_rows.append(row)

    return summarize_fold_metrics(fold_rows, ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]), pd.DataFrame(pred_rows), None


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: str, obj: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def split_metrics_tables(metrics_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    retrieval_cols = ["system", "Recall@5", "Recall@5_std", "MRR", "MRR_std", "nDCG", "nDCG_std", "retrieval_evaluated", "retrieval_evaluated_std"]
    classification_cols = ["system", "accuracy", "accuracy_std", "precision", "precision_std", "recall", "recall_std", "f1", "f1_std", "auroc", "auroc_std", "auprc", "auprc_std"]

    metrics_reset = metrics_df.reset_index()
    retrieval_df = metrics_reset[metrics_reset["system"].str.contains("retrieval")][retrieval_cols].copy()
    classification_df = metrics_reset[~metrics_reset["system"].str.contains("retrieval")][classification_cols].copy()
    return retrieval_df, classification_df


def main() -> None:
    load_default_env_files()

    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--policy_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dense_model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--chunk_overlap", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--llm_backend", choices=["none", "openai", "azure_openai"], default="none")
    parser.add_argument("--llm_model", default="gpt-4.1-mini")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    ensure_dir(args.out_dir)

    assets = load_assets(args.assets)
    effective_n_splits = resolve_n_splits(assets, args.n_splits)
    chunks = load_policy_chunks(args.policy_dir, chunk_size=args.chunk_size, overlap=args.chunk_overlap)

    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, args.dense_model)
    hybrid = HybridRetriever(bm25=bm25, dense=dense, alpha=0.5)
    rules = RuleBaseline()

    llm_rag = OptionalOpenAIRAG(args.llm_model, args.llm_backend) if args.llm_backend != "none" else None
    splitter = StratifiedKFold(n_splits=effective_n_splits, shuffle=True, random_state=args.seed)
    if effective_n_splits != args.n_splits:
        print(
            f"Adjusted n_splits from {args.n_splits} to {effective_n_splits} "
            "based on the smallest class size."
        )

    all_results = []
    system_metrics: Dict[str, Dict[str, float]] = {}

    # Retrieval metrics with fold-level summary
    for name, retriever in [("bm25_retrieval_cv", bm25), ("dense_retrieval_cv", dense), ("hybrid_retrieval_cv", hybrid)]:
        summary, pred_df = evaluate_retrieval_cv(assets, retriever, args.top_k, splitter)
        system_metrics[name] = summary
        if not pred_df.empty:
            pred_df["system"] = name
            all_results.extend(pred_df.to_dict(orient="records"))

    # Rule baseline with CV
    rules_summary, rules_pred_df = evaluate_rules_cv(assets, rules, splitter)
    system_metrics["rules_cv"] = rules_summary
    all_results.extend(rules_pred_df.to_dict(orient="records"))

    # TF-IDF classifier with CV
    tfidf_summary, tfidf_pred_df = evaluate_tfidf_cv(assets, splitter)
    system_metrics["tfidf_cv"] = tfidf_summary
    all_results.extend(tfidf_pred_df.to_dict(orient="records"))

    # Retrieval-augmented TF-IDF/LR with CV
    for name, retriever in [("bm25_rag_cv", bm25), ("dense_rag_cv", dense), ("hybrid_rag_cv", hybrid)]:
        summary, pred_df = evaluate_rag_cv(
            assets=assets,
            retriever=retriever,
            top_k=args.top_k,
            n_splits=effective_n_splits,
            seed=args.seed,
        )
        system_metrics[name] = summary
        if not pred_df.empty:
            pred_df["system"] = name
            all_results.extend(pred_df.to_dict(orient="records"))

    # Optional LLM RAG; skip failed API paths instead of reporting them as model outputs.
    if llm_rag is not None:
        llm_summary, llm_pred_df, llm_error = evaluate_llm_rag_cv(assets, hybrid, llm_rag, args.top_k, splitter)
        if llm_error is None and llm_summary is not None:
            system_metrics["llm_hybrid_rag_cv"] = llm_summary
            all_results.extend(llm_pred_df.to_dict(orient="records"))
        else:
            system_metrics["llm_hybrid_rag_cv"] = {"status": "skipped", "error": llm_error or "unknown-error"}

    metrics_df = pd.DataFrame.from_dict(system_metrics, orient="index")
    metrics_df.index.name = "system"
    metrics_reset = metrics_df.reset_index()
    retrieval_metrics_df, classification_metrics_df = split_metrics_tables(metrics_df)

    metrics_reset.to_csv(Path(args.out_dir) / "metrics_summary.csv", index=False)
    pd.DataFrame(all_results).to_csv(Path(args.out_dir) / "per_asset_predictions.csv", index=False)
    retrieval_metrics_df.to_csv(Path(args.out_dir) / "retrieval_metrics_summary.csv", index=False)
    classification_metrics_df.to_csv(Path(args.out_dir) / "classification_metrics_summary.csv", index=False)

    save_json(str(Path(args.out_dir) / "metrics_summary.json"), json.loads(metrics_reset.to_json(orient="records")))
    save_json(
        str(Path(args.out_dir) / "retrieval_metrics_summary.json"),
        json.loads(retrieval_metrics_df.to_json(orient="records")),
    )
    save_json(
        str(Path(args.out_dir) / "classification_metrics_summary.json"),
        json.loads(classification_metrics_df.to_json(orient="records")),
    )

    print("Saved:")
    print(Path(args.out_dir) / "metrics_summary.csv")
    print(Path(args.out_dir) / "retrieval_metrics_summary.csv")
    print(Path(args.out_dir) / "classification_metrics_summary.csv")
    print(Path(args.out_dir) / "per_asset_predictions.csv")
    print("\nSummary:\n")
    print(metrics_df.fillna("NA").to_string())


if __name__ == "__main__":
    main()
