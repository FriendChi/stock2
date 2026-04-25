from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sqlite3

import pandas as pd


def _serialize_signature(signature):
    return json.dumps(list(signature), ensure_ascii=False, separators=(",", ":"))


def _normalize_summary_value(value):
    if pd.isna(value):
        return None
    return float(value)


def _clone_summary(summary):
    cloned_summary = dict(summary)
    cloned_summary["candidate_label_list"] = [str(item) for item in cloned_summary.get("candidate_label_list", [])]
    return cloned_summary


def build_forward_selection_cache_path(
    output_dir,
    fund_code,
    resolved_stability_analysis_path,
    resolved_preprocess_path,
    target_nav_column,
    selected_candidate_label_list,
    walk_forward_config,
    data_split_dict,
    ic_aggregation_config,
    dedup_root_topk,
):
    resolved_preprocess_path = Path(resolved_preprocess_path)
    preprocess_stat = resolved_preprocess_path.stat()
    fingerprint_payload = {
        "stability_analysis_path": str(Path(resolved_stability_analysis_path).resolve()),
        "preprocess_path": str(resolved_preprocess_path.resolve()),
        "preprocess_mtime_ns": int(preprocess_stat.st_mtime_ns),
        "preprocess_size": int(preprocess_stat.st_size),
        "target_nav_column": str(target_nav_column),
        "selected_candidate_label_list": [str(item) for item in selected_candidate_label_list],
        "walk_forward_config": dict(walk_forward_config),
        "data_split_dict": dict(data_split_dict),
        "ic_aggregation_config": dict(ic_aggregation_config),
        "dedup_root_topk": int(dedup_root_topk),
    }
    fingerprint_text = json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()[:16]
    return Path(output_dir) / f"forward_selection_cache_{str(fund_code).zfill(6)}_{fingerprint}.sqlite"


class ForwardSelectionSummaryCache:
    def __init__(self, sqlite_path, memory_cache_size=5000, flush_every=200):
        self.sqlite_path = Path(sqlite_path)
        self.memory_cache_size = max(int(memory_cache_size), 0)
        self.flush_every = max(int(flush_every), 1)
        self._memory_cache = OrderedDict()
        self._pending_write_count = 0
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.sqlite_path))
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS forward_selection_summary_cache (
                signature TEXT PRIMARY KEY,
                candidate_label_list_json TEXT NOT NULL,
                factor_count INTEGER NOT NULL,
                train_spearman_ic_mean REAL,
                train_spearman_icir REAL,
                valid_spearman_ic_mean REAL,
                valid_spearman_icir REAL,
                step INTEGER,
                has_valid_metrics INTEGER NOT NULL
            )
            """
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _set_memory_cache(self, signature_key, summary):
        if self.memory_cache_size <= 0:
            return
        cached_summary = _clone_summary(summary)
        if signature_key in self._memory_cache:
            self._memory_cache.move_to_end(signature_key)
        self._memory_cache[signature_key] = cached_summary
        while len(self._memory_cache) > self.memory_cache_size:
            self._memory_cache.popitem(last=False)

    def get(self, signature):
        signature_key = _serialize_signature(signature)
        cached_summary = self._memory_cache.get(signature_key)
        if cached_summary is not None:
            self._memory_cache.move_to_end(signature_key)
            return _clone_summary(cached_summary)

        row = self._connection.execute(
            """
            SELECT
                candidate_label_list_json,
                factor_count,
                train_spearman_ic_mean,
                train_spearman_icir,
                valid_spearman_ic_mean,
                valid_spearman_icir,
                step,
                has_valid_metrics
            FROM forward_selection_summary_cache
            WHERE signature = ?
            """,
            (signature_key,),
        ).fetchone()
        if row is None:
            return None
        summary = {
            "candidate_label_list": json.loads(row[0]),
            "factor_count": int(row[1]),
            "train_spearman_ic_mean": float("nan") if row[2] is None else float(row[2]),
            "train_spearman_icir": float("nan") if row[3] is None else float(row[3]),
            "step": None if row[6] is None else int(row[6]),
        }
        if bool(row[7]):
            summary["valid_spearman_ic_mean"] = float("nan") if row[4] is None else float(row[4])
            summary["valid_spearman_icir"] = float("nan") if row[5] is None else float(row[5])
        self._set_memory_cache(signature_key=signature_key, summary=summary)
        return _clone_summary(summary)

    def set(self, signature, summary):
        signature_key = _serialize_signature(signature)
        normalized_summary = {
            "candidate_label_list": [str(item) for item in summary["candidate_label_list"]],
            "factor_count": int(summary["factor_count"]),
            "train_spearman_ic_mean": _normalize_summary_value(summary.get("train_spearman_ic_mean")),
            "train_spearman_icir": _normalize_summary_value(summary.get("train_spearman_icir")),
            "step": None if summary.get("step") is None else int(summary["step"]),
        }
        has_valid_metrics = "valid_spearman_ic_mean" in summary and "valid_spearman_icir" in summary
        if has_valid_metrics:
            normalized_summary["valid_spearman_ic_mean"] = _normalize_summary_value(summary.get("valid_spearman_ic_mean"))
            normalized_summary["valid_spearman_icir"] = _normalize_summary_value(summary.get("valid_spearman_icir"))
        self._set_memory_cache(signature_key=signature_key, summary=normalized_summary)

        self._connection.execute(
            """
            INSERT OR REPLACE INTO forward_selection_summary_cache (
                signature,
                candidate_label_list_json,
                factor_count,
                train_spearman_ic_mean,
                train_spearman_icir,
                valid_spearman_ic_mean,
                valid_spearman_icir,
                step,
                has_valid_metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signature_key,
                json.dumps(normalized_summary["candidate_label_list"], ensure_ascii=False, separators=(",", ":")),
                normalized_summary["factor_count"],
                normalized_summary["train_spearman_ic_mean"],
                normalized_summary["train_spearman_icir"],
                normalized_summary.get("valid_spearman_ic_mean"),
                normalized_summary.get("valid_spearman_icir"),
                normalized_summary["step"],
                int(has_valid_metrics),
            ),
        )
        self._pending_write_count += 1
        if self._pending_write_count >= self.flush_every:
            self.flush()

    def flush(self):
        if self._pending_write_count <= 0:
            return
        self._connection.commit()
        self._pending_write_count = 0

    def close(self):
        if self._connection is None:
            return
        self.flush()
        self._connection.close()
        self._connection = None
