r"""
BioRED RRF 混合检索测试脚本

目的：测试把 chunk 检索层的召回排序从"纯 Embedding 余弦相似度"换成
     "BM25 排名 + Embedding 排名 → RRF 倒数排名融合(k=60)" 后的效果。

设计（经需求确认）：
1. 仅替换 ChunkRetriever 召回排序这一段（原：向量相似度排序取top-20；
   新：RRF融合排序取top-20），gte-rerank-v2 精排及其 0.7/0.3 组合公式保持不变
2. BM25 查询词只用两个实体名（MeSH规范化后的），不带自然语言虚词；
   Embedding 查询保持原样（自然语言问句）
3. 抽样：按金标准标签分层随机抽取 N 对（seed 固定可复现），清单存 manifest 供复跑复用
4. 通过猴子补丁注入，不修改任何主程序代码

运行方式：
    python test_biored_rrf_hybrid.py 3 hybrid     # 跑3对（冒烟测试）
    python test_biored_rrf_hybrid.py 896 hybrid   # 混合臂全量
    python test_biored_rrf_hybrid.py 896 baseline # 对照臂全量
    python test_biored_rrf_hybrid.py report       # 只出报告不跑样本

输出（results/ 下，路径可用环境变量 GRACE_RESULTS_DIR 覆盖）：
    manifest_n896_seed42.json  分层抽样清单（两臂复用同一份）
    baseline_results.csv / hybrid_results.csv   每样本判断结果
    comparison_vs_old.csv  与参考CSV同对的对照（需可选提供参考CSV）
    comparison_report.txt  最终对比报告（P/R/F1）
"""

import os
import sys

# 设置UTF-8编码（Windows兼容）
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import re
import csv
import json
import random
import time
import asyncio
import numpy as np
from pathlib import Path

# 添加主程序目录到路径
LIGHTRAG_CORE_PATH = os.getenv("LIGHTRAG_CORE_PATH", str(Path(__file__).resolve().parent))
if LIGHTRAG_CORE_PATH not in sys.path:
    sys.path.insert(0, LIGHTRAG_CORE_PATH)

from lightrag_core import enhanced_query_entity_relation, initialize_rag, qwen_llm_func
import syRACT_dynamic_rag_with_omop as syRACT

# ============== 配置区域 ==============

N_SAMPLES = 100         # 抽取样本数
SEED = 42               # 分层随机抽样种子
RRF_K = 60              # RRF 常数 k
RECALL_K = 20           # RRF 融合后召回数量（与原向量召回 recall_k 一致）
TOP_K = 5               # 最终返回chunk数（与主程序一致）

CSV_PATH = os.getenv("BIORED_CSV", str(Path(__file__).resolve().parent / "biored_test_samples.csv"))
BASE_RESULTS = os.getenv("BASE_RESULTS", str(Path(__file__).resolve().parent / "biored_test_results.csv"))
CACHE_DIR = Path(os.getenv("GRACE_RESULTS_DIR", str(Path(__file__).resolve().parent / "results")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RETRIEVAL_LOG = []  # 本次阶段每次检索的明细


# ============== BioRED 读取与类型映射（与 test_biored.py 一致） ==============

def read_biored_csv(csv_path: str, n_limit: int = None) -> list:
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if n_limit and i >= n_limit:
                break
            samples.append({
                'pmid': row.get('pmid', ''),
                'entity1_text': row.get('entity1_text', ''),
                'entity2_text': row.get('entity2_text', ''),
                'entity1_type': row.get('entity1_type', ''),
                'entity2_type': row.get('entity2_type', ''),
                'abstract': row.get('abstract', ''),
                'true_label': row.get('true_label', '').lower() == 'yes',
                'relation_type': row.get('relation_type', '')
            })
    return samples


def map_entity_type(biored_type: str) -> str:
    type_mapping = {
        'GeneOrGeneProduct': 'GeneOrProtein',
        'ChemicalEntity': 'ChemicalOrDrug',
        'DiseaseOrPhenotypicFeature': 'Disease',
        'SequenceVariant': 'GeneOrProtein',
        'Complex': 'GeneOrProtein',
        'Cell': 'GeneOrProtein'
    }
    return type_mapping.get(biored_type, biored_type)


def load_stratified_samples(n: int, seed: int = SEED) -> list:
    """按金标准标签分层随机抽取 n 对（文本级键去重，seed固定可复现）。

    清单落盘 manifest_n{n}_seed{seed}.json，已存在则直接加载，
    保证 baseline/hybrid 两臂与断点复跑使用同一份样本。
    """
    manifest = CACHE_DIR / f"manifest_n{n}_seed{seed}.json"
    if manifest.exists():
        return json.loads(manifest.read_text(encoding='utf-8'))

    seen = set()
    pool = {'yes': [], 'no': []}
    for s in read_biored_csv(CSV_PATH):
        key = (s['pmid'], s['entity1_text'], s['entity2_text'], s['relation_type'])
        if key in seen:
            continue
        seen.add(key)
        pool['yes' if s['true_label'] else 'no'].append(s)

    rng = random.Random(seed)
    n_total = len(pool['yes']) + len(pool['no'])
    if n >= n_total:
        # n 覆盖全总体：直接全量（去重后全部唯一对）
        picked = pool['yes'] + pool['no']
    else:
        n_yes = round(n * len(pool['yes']) / n_total)
        picked = rng.sample(pool['yes'], n_yes) + rng.sample(pool['no'], n - n_yes)
    manifest.write_text(json.dumps(picked, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"[抽样] 清单已保存: {manifest}")
    return picked


# ============== RRF 混合检索核心 ==============

def _tokenize(text: str) -> list:
    """小写 + 正则切词（不做词干化，避免医学词形被破坏）"""
    return re.findall(r'[a-z0-9]+', text.lower())


def _ranks_desc(scores) -> np.ndarray:
    """分数→名次（1-based，分数高名次前，平分时按原顺序）"""
    order = np.argsort(-np.asarray(scores, dtype=float), kind='stable')
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


async def _rerank_stage_async(self, query: str, recall_chunks: list, top_k: int) -> list:
    """精排阶段（异步版）：复制原 retrieve 的 Rerank 逻辑，保持组合公式不变"""
    if not self.enable_rerank or not self.rerank_func or len(recall_chunks) <= top_k:
        print(f"    [Hybrid] 无需rerank，直接返回 RRF top-{top_k}")
        return recall_chunks[:top_k]

    try:
        print(f"    [Hybrid] Rerank精排: {len(recall_chunks)} chunks → top-{top_k}")
        documents = [item['chunk_text'][:1500] for item in recall_chunks]
        rerank_results = await self.rerank_func(query, documents)

        rerank_scores = {}
        for result in rerank_results:
            orig_idx = result['index']
            score = result['relevance_score']
            if 0 <= orig_idx < len(recall_chunks):
                rerank_scores[orig_idx] = score

        for idx, item in enumerate(recall_chunks):
            rerank_score = rerank_scores.get(idx, 0.0)
            item['rerank_score'] = rerank_score
            item['combined_score'] = 0.7 * rerank_score + 0.3 * item['similarity']

        recall_chunks.sort(key=lambda x: x['combined_score'], reverse=True)

        print(f"    [Hybrid] Rerank完成，Top-{top_k} 结果:")
        for i, item in enumerate(recall_chunks[:top_k]):
            pmid = item['metadata'].get('pmid', 'N/A')
            print(f"      [{i+1}] PMID:{pmid} | RRF:{item.get('rrf_score', 0):.5f} "
                  f"(BM25#{int(item.get('bm25_rank', 0))}/Emb#{int(item.get('emb_rank', 0))}) "
                  f"| 向量:{item['similarity']:.3f} | Rerank:{item.get('rerank_score', 0.0):.3f} "
                  f"| 混合:{item.get('combined_score', 0.0):.3f}")
        return recall_chunks[:top_k]
    except Exception as e:
        print(f"    [Hybrid] Rerank失败: {e}，回退到RRF排序")
        return recall_chunks[:top_k]


async def hybrid_retrieve(self, query: str, top_k: int = TOP_K, recall_k: int = RECALL_K) -> list:
    """混合检索：BM25排名 + Embedding排名 → RRF(k=60) → top recall_k → 原rerank精排

    仅替换原 retrieve 的召回排序（原为向量相似度排序），
    精排阶段复制原逻辑，保证只有"检索方式"这一个变量。
    """
    # BM25 查询词：只用两个实体名（MeSH规范化名，由 build 补丁存到 retriever 上）
    entities = getattr(self, '_query_entities', None)
    if entities and entities[0] and entities[1]:
        bm25_query = f"{entities[0]} {entities[1]}"
    else:
        bm25_query = query
        print(f"    [Hybrid] 警告：未获取到实体名，BM25退化为完整查询句")

    n = len(self.chunks)
    if n == 0:
        return []

    # === Embedding 分数（全量chunk） ===
    query_emb = await syRACT.get_qwen_embeddings([query])
    query_vec = query_emb[0]
    emb_scores = [float(syRACT.cosine_similarity(query_vec, emb)) for emb in self.embeddings]

    # === BM25 分数（全量chunk） ===
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [_tokenize(c) for c in self.chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(_tokenize(bm25_query))

    # === 分数→名次→RRF融合（只看名次，抛弃原始分数） ===
    emb_ranks = _ranks_desc(emb_scores)
    bm25_ranks = _ranks_desc(bm25_scores)
    rrf_scores = 1.0 / (RRF_K + bm25_ranks) + 1.0 / (RRF_K + emb_ranks)

    items = []
    for idx in range(n):
        items.append({
            'idx': idx,
            'chunk_text': self.chunks[idx],
            'similarity': emb_scores[idx],
            'bm25_rank': float(bm25_ranks[idx]),
            'emb_rank': float(emb_ranks[idx]),
            'rrf_score': float(rrf_scores[idx]),
            'metadata': self.chunk_metadata[idx]
        })
    items.sort(key=lambda x: x['rrf_score'], reverse=True)
    recall_chunks = items[:recall_k] if len(items) > recall_k else items

    # === 打印RRF融合明细 ===
    print(f"    [Hybrid] RRF融合(k={RRF_K}): {n} chunks → 召回top-{len(recall_chunks)}")
    print(f"    [Hybrid] BM25查询词: '{bm25_query}'")
    for i, item in enumerate(recall_chunks[:8]):
        pmid = item['metadata'].get('pmid', 'N/A')
        print(f"      #{i+1} PMID:{pmid} | RRF:{item['rrf_score']:.5f} "
              f"= 1/{RRF_K}+BM25名次{int(item['bm25_rank'])} + 1/{RRF_K}+Emb名次{int(item['emb_rank'])}")

    # === 精排（与原版一致的rerank逻辑） ===
    final = await _rerank_stage_async(self, query, recall_chunks, top_k)

    RETRIEVAL_LOG.append({
        'query': bm25_query,
        'recall': [
            {
                'pmid': it['metadata'].get('pmid', ''),
                'chunk_id': it['metadata'].get('chunk_id', ''),
                'bm25_rank': int(it['bm25_rank']),
                'emb_rank': int(it['emb_rank']),
                'rrf_score': round(it['rrf_score'], 6),
                'similarity': round(it['similarity'], 4),
                'rerank_score': round(it.get('rerank_score', 0.0), 4),
                'combined_score': round(it.get('combined_score', 0.0), 4)
            }
            for it in final
        ]
    })
    return final


async def baseline_log_retrieve(self, query: str, top_k: int = TOP_K, recall_k: int = RECALL_K) -> list:
    """原版检索 + 记录明细（逻辑完全不变，仅记录返回的top-k）"""
    result = await ORIGINAL_RETRIEVE(self, query, top_k=top_k, recall_k=recall_k)
    RETRIEVAL_LOG.append({
        'query': query,
        'recall': [
            {
                'pmid': it['metadata'].get('pmid', ''),
                'chunk_id': it['metadata'].get('chunk_id', ''),
                'bm25_rank': '',
                'emb_rank': '',
                'rrf_score': '',
                'similarity': round(it.get('similarity', 0.0), 4),
                'rerank_score': round(it.get('rerank_score', 0.0), 4),
                'combined_score': round(it.get('combined_score', 0.0), 4)
            }
            for it in result
        ]
    })
    return result


ORIGINAL_RETRIEVE = syRACT.ChunkRetriever.retrieve
ORIGINAL_BUILD = syRACT.EnhancedDynamicRAG.build_rag_for_query


def apply_patches(mode: str):
    """mode: 'baseline' | 'hybrid'。实体名传递补丁所有模式都打。"""
    async def build_patch(self, entity1, entity2, *args, **kwargs):
        self.retriever._query_entities = (entity1, entity2)
        return await ORIGINAL_BUILD(self, entity1, entity2, *args, **kwargs)

    syRACT.EnhancedDynamicRAG.build_rag_for_query = build_patch

    if mode == 'hybrid':
        syRACT.ChunkRetriever.retrieve = hybrid_retrieve
        print("[Patch] 已注入混合检索（BM25+Embedding→RRF→rerank）")
    else:
        syRACT.ChunkRetriever.retrieve = baseline_log_retrieve
        print("[Patch] 已注入原版检索（仅记录明细，逻辑不变）")


def restore_patches():
    syRACT.ChunkRetriever.retrieve = ORIGINAL_RETRIEVE
    syRACT.EnhancedDynamicRAG.build_rag_for_query = ORIGINAL_BUILD


# ============== 结果写出 ==============

RESULT_FIELDS = ['mode', 'idx', 'pmid', 'entity_a', 'entity_b', 'true_label',
                 'predicted', 'correct', 'confidence', 'llm_score', 'graph_score',
                 'multi_hop_path', 'relation_type', 'elapsed_sec',
                 'graph_pmids', 'graph_source_mode']


def append_result(result_csv: Path, row: dict):
    is_new = not result_csv.exists()
    with open(result_csv, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def flush_retrieval_log(retrieval_csv: Path):
    if not RETRIEVAL_LOG:
        return
    is_new = not retrieval_csv.exists()
    with open(retrieval_csv, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(['query', 'rank', 'pmid', 'chunk_id', 'bm25_rank',
                             'emb_rank', 'rrf_score', 'similarity', 'rerank_score', 'combined_score'])
        for entry in RETRIEVAL_LOG:
            for r, item in enumerate(entry['recall'], 1):
                writer.writerow([entry['query'], r, item['pmid'], item['chunk_id'],
                                 item['bm25_rank'], item['emb_rank'], item['rrf_score'],
                                 item['similarity'], item['rerank_score'], item['combined_score']])
    RETRIEVAL_LOG.clear()


# ============== 测试主流程 ==============

async def test_single_sample(sample: dict, rag, llm_func, idx: int, total: int, mode: str) -> dict:
    pmid = sample['pmid']
    entity_a = sample['entity1_text']
    entity_b = sample['entity2_text']
    entity_a_type = map_entity_type(sample['entity1_type'])
    entity_b_type = map_entity_type(sample['entity2_type'])
    abstract = sample['abstract']
    true_label = sample['true_label']

    print(f"\n{'='*60}")
    print(f"[{mode} 样本 {idx}/{total}] PMID: {pmid}")
    print(f"实体A: {entity_a} ({entity_a_type})")
    print(f"实体B: {entity_b} ({entity_b_type})")
    print(f"金标准: {'YES' if true_label else 'NO'} ({sample['relation_type']})")
    print(f"{'='*60}")

    t0 = time.time()
    result = await enhanced_query_entity_relation(
        rag=rag,
        entity_a=entity_a,
        entity_b=entity_b,
        llm_func=llm_func,
        abstract=abstract,
        enable_rerank=True,
        entity_a_type=entity_a_type,
        entity_b_type=entity_b_type
    )
    elapsed = time.time() - t0

    if 'has_relation' not in result or result['has_relation'] is None:
        # 无效输出与合法NO必须区分：缺字段按失败处理（中断运行，断点续跑补齐），不得静默记为no
        raise ValueError(f"验证结果缺少has_relation字段: keys={sorted(result.keys())}")
    confidence = float(result.get('confidence', float('nan')))
    if not (0.0 <= confidence <= 1.0):
        # confidence越界（含NaN，NaN比较恒False）说明上游解析失败，按失败处理
        raise ValueError(f"confidence 越界[0,1]: {confidence}")
    predicted = bool(result['has_relation'])
    row = {
        'mode': mode,
        'idx': idx,
        'pmid': pmid,
        'entity_a': entity_a,
        'entity_b': entity_b,
        'true_label': 'yes' if true_label else 'no',
        'predicted': 'yes' if predicted else 'no',
        'correct': int(predicted == true_label),
        'confidence': round(confidence, 4),
        'llm_score': round(float(result.get('llm_score', 0.0)), 4),
        'graph_score': round(float(result.get('graph_score', 0.0)), 4),
        'multi_hop_path': ' -> '.join(result.get('multi_hop_path', []) or []),
        'relation_type': sample['relation_type'],
        'elapsed_sec': round(elapsed, 1),
        'graph_pmids': ';'.join(result.get('graph_build_sources', []) or []),
        'graph_source_mode': result.get('graph_data_quality', 'unknown')
    }
    mark = '✓' if row['correct'] else '✗'
    print(f"\n[{mode}] 判断: {'YES' if predicted else 'NO'} | 真实: {'YES' if true_label else 'NO'} "
          f"| {mark} | confidence={row['confidence']:.2f} | graph={row['graph_score']:.2f} | {elapsed:.0f}s")
    return row


async def run_phase(mode: str, samples: list, rag, llm_func):
    result_csv = CACHE_DIR / f"{mode}_results.csv"
    retrieval_csv = CACHE_DIR / f"{mode}_retrieval.csv"
    # 每次调用完整重跑该臂：先重置结果文件，确保输出不含历史运行残留（无断点续跑逻辑）
    for stale in (result_csv, retrieval_csv):
        if stale.exists():
            stale.unlink()
    print(f"\n{'#'*70}")
    print(f"# 阶段: {mode.upper()} | 样本 {len(samples)}")
    print(f"{'#'*70}")

    apply_patches(mode)
    try:
        for i, sample in enumerate(samples, 1):
            row = await test_single_sample(sample, rag, llm_func, i, len(samples), mode)
            append_result(result_csv, row)
            flush_retrieval_log(retrieval_csv)
    finally:
        restore_patches()


def _prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _load_arm_stats(mode: str):
    """读取单臂结果，返回统计与全部行"""
    f = CACHE_DIR / f"{mode}_results.csv"
    if not f.exists():
        return None, []
    with open(f, 'r', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    if not n:
        return {'n': 0}, []
    pos = [r for r in rows if r['true_label'] == 'yes']
    neg = [r for r in rows if r['true_label'] == 'no']
    tp = sum(1 for r in pos if r['predicted'] == 'yes')
    fn = len(pos) - tp
    tn = sum(1 for r in neg if r['predicted'] == 'no')
    fp = len(neg) - tn
    p, rec, f1 = _prf(tp, fp, fn)
    stats = {
        'n': n,
        'acc': sum(int(r['correct']) for r in rows) / n,
        'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp, 'p': p, 'recall': rec, 'f1': f1,
        'avg_conf': sum(float(r['confidence']) for r in rows) / n,
        'avg_graph': sum(float(r['graph_score']) for r in rows) / n,
        'avg_time': sum(float(r['elapsed_sec']) for r in rows) / n
    }
    return stats, rows


def write_report():
    lines = ["=" * 60, "RRF 混合检索 F1 报告", "=" * 60]
    stats = {}
    for mode in ['baseline', 'hybrid']:
        s, _ = _load_arm_stats(mode)
        if s is None:
            continue
        stats[mode] = s
        name = '原版Embedding检索' if mode == 'baseline' else 'RRF混合检索'
        lines.append(f"\n【{name}】 n={s['n']}")
        lines.append(f"  准确率: {s['acc']*100:.1f}%")
        lines.append(f"  混淆矩阵: TP={s['tp']} FN={s['fn']} TN={s['tn']} FP={s['fp']}")
        lines.append(f"  P={s['p']:.4f}  R={s['recall']:.4f}  F1={s['f1']:.4f}（正类=YES）")
        lines.append(f"  平均confidence: {s['avg_conf']:.3f} | 平均graph_score: {s['avg_graph']:.3f} "
                     f"| 平均耗时: {s['avg_time']:.0f}s")

    if 'baseline' in stats and 'hybrid' in stats:
        b, h = stats['baseline'], stats['hybrid']
        lines.append(f"\n【两臂对比】 F1: {b['f1']:.4f} → {h['f1']:.4f} "
                     f"({'+' if h['f1'] >= b['f1'] else ''}{(h['f1']-b['f1'])*100:.2f} 个百分点) | "
                     f"准确率: {b['acc']*100:.1f}% → {h['acc']*100:.1f}%")
        with open(CACHE_DIR / 'baseline_results.csv', encoding='utf-8') as fh:
            bmap = {(r['pmid'], r['entity_a'], r['entity_b']): r
                    for r in csv.DictReader(fh)}
        with open(CACHE_DIR / 'hybrid_results.csv', encoding='utf-8') as fh:
            hmap = {(r['pmid'], r['entity_a'], r['entity_b']): r
                    for r in csv.DictReader(fh)}

        # 两臂分歧的 McNemar 精确检验（论文口径）
        b_only = sum(1 for k in set(bmap) & set(hmap)
                     if bmap[k]['predicted'] == 'yes' and hmap[k]['predicted'] != 'yes')
        h_only = sum(1 for k in set(bmap) & set(hmap)
                     if hmap[k]['predicted'] == 'yes' and bmap[k]['predicted'] != 'yes')
        n_disc = b_only + h_only
        if n_disc:
            from math import comb
            p = sum(comb(n_disc, i) for i in range(0, min(b_only, h_only) + 1)) \
                / sum(comb(n_disc, i) for i in range(0, n_disc + 1)) * 2
            lines.append(f"  McNemar 精确检验(双侧): baseline独判YES {b_only} 对, "
                         f"hybrid独判YES {h_only} 对, p={min(p, 1.0):.4f}")

        lines.append("\n【逐样本对比】")
        lines.append(f"  {'PMID':<12} {'实体':<40} {'真实':<5} {'原版':<5} {'混合':<5}")
        for key, br in bmap.items():
            hr = hmap.get(key)
            if not hr:
                continue
            ent = f"{br['entity_a'][:18]}~{br['entity_b'][:18]}"
            flip = ' ←翻转' if br['predicted'] != hr['predicted'] else ''
            lines.append(f"  {br['pmid']:<12} {ent:<40} {br['true_label']:<5} "
                         f"{br['predicted']:<5}({float(br['confidence']):.2f}) "
                         f"{hr['predicted']:<5}({float(hr['confidence']):.2f}){flip}")

    report = CACHE_DIR / "comparison_report.txt"
    report.write_text('\n'.join(lines), encoding='utf-8')
    print('\n' + '\n'.join(lines))
    print(f"\n[报告] 已保存: {report}")


def compare_with_old(mode: str = 'hybrid'):
    """与参考CSV同对结果对照（参考值仅作对照参考）。"""
    import pandas as pd
    if not Path(BASE_RESULTS).exists():
        print(f"[对照] 未提供 BASE_RESULTS（{BASE_RESULTS}），跳过旧CSV对照。")
        return
    _, rows = _load_arm_stats(mode)
    if not rows:
        print(f"[对照] {mode} 无有效结果，跳过旧CSV对照")
        return

    old = pd.read_csv(BASE_RESULTS, encoding='latin-1', on_bad_lines='skip')
    old = old.drop_duplicates(subset=['pmid', 'entity_a', 'entity_b'], keep='last')
    omap = {}
    for _, r in old.iterrows():
        if r['llm_has_relation'] in ('YES', 'NO'):
            omap[(str(r['pmid']), str(r['entity_a']), str(r['entity_b']))] = r

    out_rows, missing = [], 0
    for nr in rows:
        orow = omap.get((nr['pmid'], nr['entity_a'], nr['entity_b']))
        old_pred = ''
        old_conf = old_graph = ''
        if orow is None:
            missing += 1
        else:
            old_pred = 'yes' if orow['llm_has_relation'] == 'YES' else 'no'
            old_conf = float(orow['llm_confidence']) if pd.notna(orow['llm_confidence']) else ''
            old_graph = float(orow['graph_score']) if pd.notna(orow['graph_score']) else ''
        out_rows.append({
            'pmid': nr['pmid'], 'entity_a': nr['entity_a'], 'entity_b': nr['entity_b'],
            'true_label': nr['true_label'], 'relation_type': nr['relation_type'],
            'old_predicted': old_pred, 'old_confidence': old_conf, 'old_graph_score': old_graph,
            'new_predicted': nr['predicted'], 'new_confidence': nr['confidence'],
            'new_graph_score': nr['graph_score'],
            'old_correct': int(old_pred == nr['true_label']) if old_pred else '',
            'new_correct': nr['correct'],
        })

    def _arm_metrics(key_pred, key_correct):
        valid = [r for r in out_rows if r[key_pred]]
        pos = [r for r in valid if r['true_label'] == 'yes']
        neg = [r for r in valid if r['true_label'] == 'no']
        tp = sum(1 for r in pos if r[key_pred] == 'yes')
        fn = len(pos) - tp
        tn = sum(1 for r in neg if r[key_pred] == 'no')
        fp = len(neg) - tn
        p, rec, f1 = _prf(tp, fp, fn)
        return {'n': len(valid), 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
                'p': p, 'recall': rec, 'f1': f1,
                'acc': sum(int(r[key_correct]) for r in valid) / len(valid) if valid else 0}

    om, nm = _arm_metrics('old_predicted', 'old_correct'), _arm_metrics('new_predicted', 'new_correct')

    lines = ["=" * 60,
             f"RRF混合检索 vs 参考CSV对照（同{len(out_rows)}对，参考值仅作对照）",
             "=" * 60,
             f"\n【旧结果（参考）】 n={om['n']}  P={om['p']:.4f} R={om['recall']:.4f} F1={om['f1']:.4f} "
             f"TP={om['tp']} FN={om['fn']} TN={om['tn']} FP={om['fp']}",
             f"\n【RRF混合（当前代码）】 n={nm['n']}  P={nm['p']:.4f} R={nm['recall']:.4f} F1={nm['f1']:.4f} "
             f"TP={nm['tp']} FN={nm['fn']} TN={nm['tn']} FP={nm['fp']}",
             f"\n【F1变化】 {om['f1']:.4f} → {nm['f1']:.4f} ({(nm['f1']-om['f1'])*100:+.2f} 个百分点)",
             f"【准确率变化】 {om['acc']*100:.1f}% → {nm['acc']*100:.1f}%"]

    if missing:
        lines.append(f"\n[注意] {missing} 对在旧CSV中未找到（无旧结果对照）")

    fixed = [r for r in out_rows if r['old_predicted'] and r['old_correct'] == 0 and int(r['new_correct']) == 1]
    broken = [r for r in out_rows if r['old_predicted'] and r['old_correct'] == 1 and int(r['new_correct']) == 0]
    lines.append(f"\n【逐对翻转】 旧错→新对: {len(fixed)} 对 | 旧对→新错: {len(broken)} 对")
    for tag, group in (('旧错→新对', fixed), ('旧对→新错', broken)):
        for r in group:
            lines.append(f"  [{tag}] PMID:{r['pmid']} {r['entity_a']} <-> {r['entity_b']} "
                         f"真实:{r['true_label']} 旧:{r['old_predicted']} → 新:{r['new_predicted']}")

    out_csv = CACHE_DIR / "comparison_vs_old.csv"
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    lines.append(f"\n[明细] 已保存: {out_csv}")

    report = CACHE_DIR / "comparison_vs_old.txt"
    report.write_text('\n'.join(lines), encoding='utf-8')
    print('\n' + '\n'.join(lines))
    print(f"\n[对照报告] 已保存: {report}")


async def main():
    # 解析参数：数字=样本数，字符串=阶段（baseline/hybrid/report）
    n_limit = N_SAMPLES
    phases = ['hybrid']
    for arg in sys.argv[1:]:
        if arg.isdigit():
            n_limit = int(arg)
        elif arg in ('baseline', 'hybrid', 'report'):
            phases = [arg]

    if phases and phases[0] == 'report':
        write_report()
        compare_with_old()
        return

    samples = load_stratified_samples(n_limit)
    n_yes = sum(1 for s in samples if s['true_label'])
    print(f"[样本] 分层随机抽取 {len(samples)} 对（seed={SEED}，YES {n_yes} / NO {len(samples)-n_yes}）")
    for i, s in enumerate(samples, 1):
        print(f"  {i}. PMID:{s['pmid']} | {s['entity1_text']} <-> {s['entity2_text']} "
              f"| 真实: {'YES' if s['true_label'] else 'NO'}")

    print(f"\n[初始化] LightRAG + Qwen + Rerank ...")
    rag = await initialize_rag()

    for mode in phases:
        await run_phase(mode, samples, rag, qwen_llm_func)

    write_report()
    compare_with_old('hybrid' if 'hybrid' in phases else phases[0])


if __name__ == '__main__':
    asyncio.run(main())
