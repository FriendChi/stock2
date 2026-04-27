import numpy as np
import pytest
from tradition.factor_analysis.sqlite_cache import ForwardSelectionSummaryCache

def test_cache_get_set_persistence(tmp_path):
    db_path = tmp_path / "test_cache.sqlite"
    signature = ("factor1", "factor2")
    summary = {
        "candidate_label_list": ["factor1", "factor2"],
        "factor_count": 2,
        "train_spearman_ic_mean": 0.05,
        "train_spearman_icir": 1.2,
        "step": 1
    }

    # 逻辑块：写入数据并刷新到磁盘
    with ForwardSelectionSummaryCache(db_path, flush_every=1) as cache:
        cache.set(signature, summary)
    
    # 逻辑块：重新打开数据库验证持久化
    with ForwardSelectionSummaryCache(db_path) as cache:
        cached_data = cache.get(signature)
        assert cached_data is not None
        assert cached_data["candidate_label_list"] == summary["candidate_label_list"]
        assert cached_data["factor_count"] == 2
        assert np.isclose(cached_data["train_spearman_ic_mean"], 0.05)

def test_cache_memory_lru(tmp_path):
    # 逻辑块：验证内存缓存的淘汰逻辑 (size=2)
    db_path = tmp_path / "test_lru.sqlite"
    with ForwardSelectionSummaryCache(db_path, memory_cache_size=2) as cache:
        cache.set(("f1",), {"candidate_label_list": ["f1"], "factor_count": 1})
        cache.set(("f2",), {"candidate_label_list": ["f2"], "factor_count": 1})
        cache.set(("f3",), {"candidate_label_list": ["f3"], "factor_count": 1})
        
        # 此时 "f1" 应该已从内存中移除（但仍在 DB 中）
        # 注意 signature 存储在 OrderedDict 中是经过序列化的 JSON 字符串
        import json
        f1_key = json.dumps(list(("f1",)), ensure_ascii=False, separators=(",", ":"))
        assert (f1_key in cache._memory_cache) is False
        assert cache.get(("f1",)) is not None # 从 DB 重新加载
