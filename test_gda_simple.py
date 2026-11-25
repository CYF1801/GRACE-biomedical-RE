"""
GDA 测试程序——无知识图谱简化版（test_gda.py 的消融对照）

与 test_gda.py 的唯一区别：
  不构建临时知识图谱，图谱不参与判断——
  跳过 build_minimal_viable_temp_graph，验证调用时 multi_hop_path=[]、
  knowledge_graph=""、graph_score=0.0（LLM 只看到 Query Document，
  prompt 中知识图谱部分为空）。

样本一一对应（方法学配对比较的关键）：
  不做随机抽样，直接读取 test_gda.py 定稿结果 gda_test_results_gda.csv 中的
  doc_id 集合，从 GDA JSON 提取完全相同的样本。保证两程序测的是同一批样本。

运行方式: python test_gda_simple.py [--n 500]
"""

import asyncio
import csv
import json
import logging
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 设置UTF-8编码（Windows兼容）
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加主程序目录到路径
LIGHTRAG_CORE_PATH = os.getenv("LIGHTRAG_CORE_PATH", str(Path(__file__).resolve().parent))
if LIGHTRAG_CORE_PATH not in sys.path:
    sys.path.insert(0, LIGHTRAG_CORE_PATH)

from dotenv import load_dotenv
load_dotenv(os.path.join(LIGHTRAG_CORE_PATH, ".env"), override=False)
if not os.getenv("NCBI_API_KEY") and os.getenv("PUBMED_API_KEY"):
    os.environ["NCBI_API_KEY"] = os.environ["PUBMED_API_KEY"]

# 导入主程序模块（注意：不导入 build_minimal_viable_temp_graph——本版本不建图）
from lightrag_core import (
    qwen_llm_func,
    prepare_query_context,
    _llm_verify_relation_with_graph_cot,
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_EMBEDDING_MODEL,
)


# 过滤主程序已知的非致命报错日志（不改动主程序任何文件）：
# prompts_config/medical_relation_prompts.yaml 缩进损坏 -> PromptManager 加载失败
# 后回退到内置 prompt（CDR/论文实际生效版本），行为不受影响，该条 ERROR 纯属噪音。
class _KnownNonFatalErrorFilter(logging.Filter):
    def filter(self, record):
        return "Failed to load prompts from" not in record.getMessage()


logging.getLogger("lightrag").addFilter(_KnownNonFatalErrorFilter())


# ============== 配置区域（可直接修改） ==============

# GDA 数据文件路径
GDA_JSON_PATH = os.getenv("GDA_JSON_PATH", str(Path(__file__).resolve().parent / "data/GDA/test.json"))

# 缓存和结果目录
CACHE_DIR = Path(os.getenv("GRACE_RESULTS_DIR", str(Path(__file__).resolve().parent / "results")))
CACHE_DIR.mkdir(exist_ok=True)

# 参照结果文件（test_gda.py 定稿结果）：从中读取 doc_id 集合，保证样本一一对应
REFERENCE_RESULTS_CSV = CACHE_DIR / "gda_test_results_gda.csv"

# 每 N 个样本显示一次进度
CACHE_INTERVAL = 10

# ============== GDA 域约束（与 test_gda.py 逐字一致，仅注入最终验证调用） ==============

GDA_CONSTRAINT = """Entity A is ALWAYS a gene/gene product; Entity B is ALWAYS a disease.
A relation ("yes") requires the abstract to explicitly link A and B
(direction never matters): genetic association (A variants / mutations /
polymorphisms are associated with B), causal or mechanistic (A causes /
contributes to / protects from B), or biomarker (A expression / activity
levels correlate with B).
Assertions across sentence boundaries count. Animal and in vitro 
studies count as evidence.

If the abstract asserts no such A-B association, answer "no".
Apply these rules in Step 0, BEFORE the standard checks"""


async def gda_llm_func(prompt, system_prompt=None, **kwargs):
    """GDA域约束包装器：仅拦截最终关系验证调用（system_prompt含step-by-step标记），"""
    if system_prompt and 'MUST follow this EXACT step-by-step format' in system_prompt:
        prompt = GDA_CONSTRAINT + '\n\n' + prompt
    return await qwen_llm_func(prompt, system_prompt=system_prompt, **kwargs)


# ============== 双臂配置 ==============
# 与 test_gda.py 一致：臂间唯一差异即验证 prompt
ARMS = [
    {'name': 'gda', 'llm': gda_llm_func},
    {'name': 'base', 'llm': qwen_llm_func},
]


# ============== 已测试样本管理 ==============

class TestedSamplesManager:
    """已测试样本管理器（基于CSV）"""

    def __init__(self, result_csv: Path):
        self.result_csv = result_csv
        self.tested_pairs: Set[Tuple[str, str, str]] = set()  # (doc_id, entity_a, entity_b)

    def load_tested(self):
        """从结果CSV加载已测试的样本"""
        if not self.result_csv.exists():
            return

        with open(self.result_csv, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get('doc_id', ''), row.get('entity_a', ''), row.get('entity_b', ''))
                self.tested_pairs.add(key)

        print(f"[已测试样本] 从结果CSV加载了 {len(self.tested_pairs)} 个已测试样本")

    def is_tested(self, doc_id: str, entity_a: str, entity_b: str) -> bool:
        """检查是否已测试"""
        return (doc_id, entity_a, entity_b) in self.tested_pairs

    def mark_tested(self, doc_id: str, entity_a: str, entity_b: str):
        """标记为已测试"""
        self.tested_pairs.add((doc_id, entity_a, entity_b))


# ============== CSV 结果管理 ==============

class CSVResultWriter:
    """CSV结果写入器（支持追加）。列结构与 test_gda.py 相同，
    其中 graph_score 恒为 0（本版本无图谱），保留该列以兼容下游分析代码。"""

    def __init__(self, result_csv: Path):
        self.result_csv = result_csv
        self.file_exists = result_csv.exists()

        # 定义CSV列名
        self.columns = [
            'test_time', 'doc_id', 'entity_a', 'entity_b',
            'entity_a_type', 'entity_b_type',
            'true_label', 'relation_type',
            'llm_has_relation', 'llm_confidence', 'llm_score', 'graph_score',
            'is_correct', 'reason', 'error'
        ]

    def ensure_header(self):
        """确保CSV文件有表头"""
        if not self.file_exists:
            with open(self.result_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()
            self.file_exists = True

    def append_result(self, result: dict):
        """追加单条结果"""
        self.ensure_header()

        with open(self.result_csv, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            row = {
                'test_time': datetime.now().isoformat(),
                'doc_id': result.get('doc_id', ''),
                'entity_a': result.get('entity_a', ''),
                'entity_b': result.get('entity_b', ''),
                'entity_a_type': result.get('entity_a_type', ''),
                'entity_b_type': result.get('entity_b_type', ''),
                'true_label': 'YES' if result.get('true_label') else 'NO',
                'relation_type': result.get('relation_type', ''),
                'llm_has_relation': 'YES' if result.get('llm_has_relation') else 'NO',
                'llm_confidence': result.get('llm_confidence', 0.0),
                'llm_score': result.get('llm_score', 0.0),
                'graph_score': result.get('graph_score', 0.0),
                'is_correct': 'YES' if result.get('is_correct') else 'NO',
                'reason': result.get('reason', '')[:500],  # 限制长度
                'error': result.get('error', '')
            }
            writer.writerow(row)

    def get_stats(self) -> dict:
        """获取结果统计"""
        if not self.result_csv.exists():
            return {'total': 0, 'correct': 0, 'accuracy': 0.0}

        total = 0
        correct = 0

        with open(self.result_csv, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                if row.get('is_correct') == 'YES':
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return {'total': total, 'correct': correct, 'accuracy': accuracy}


# ============== GDA JSON 读取 ==============

def read_gda_json(json_path: str) -> List[dict]:
    """读取 GDA JSON 文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for idx, item in enumerate(data):
        # GDA 格式: h (head/gene), t (tail/disease), relation (1=GDA关联, 0=NR无关系)
        head_entity = item.get('h', {})
        tail_entity = item.get('t', {})

        samples.append({
            'doc_id': f"GDA_{idx}",
            'entity1_id': head_entity.get('id', ''),
            'entity2_id': tail_entity.get('id', ''),
            'entity1_text': head_entity.get('name', ''),
            'entity2_text': tail_entity.get('name', ''),
            'entity1_type': head_entity.get('type', ''),
            'entity2_type': tail_entity.get('type', ''),
            'text': item.get('token', ''),
            'true_label': item.get('relation', 0) == 1,
            'relation_type': 'Gene-DiseaseAssociation' if item.get('relation', 0) == 1 else ''
        })

    return samples


def load_reference_doc_ids(ref_csv: Path) -> Set[str]:
    """从参照结果文件（test_gda.py 定稿结果）读取 doc_id 集合。
    保证本程序测试的样本与 test_gda.py 完全一一对应（方法学配对比较的前提）。"""
    doc_ids: Set[str] = set()
    if not ref_csv.exists():
        print(f"[错误] 参照结果文件不存在: {ref_csv}")
        return doc_ids

    with open(ref_csv, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row.get('doc_id', '')
            if doc_id:
                doc_ids.add(doc_id)

    print(f"[参照样本] 从 {ref_csv.name} 读取了 {len(doc_ids)} 个 doc_id")
    return doc_ids


# ============== 类型映射 ==============

def map_gda_type(gda_type: str) -> str:
    """映射 GDA 类型到主程序类型（与 lightrag_core.MEDICAL_ENTITY_TYPES 对齐）"""
    type_mapping = {
        'Gene': 'GeneOrProtein',
        'Disease': 'Disease'
    }
    return type_mapping.get(gda_type, gda_type)


# ============== 测试函数 ==============

async def _run_arm(
    sf: dict,
    context: dict,
    arm: dict,
    tested_manager: TestedSamplesManager,
    csv_writer: CSVResultWriter,
) -> dict:
    """单臂验证：在检索上下文上直接跑最终验证调用并落盘（无图谱）。
    与 test_gda.py 的唯一差异：不建图，multi_hop_path/knowledge_graph 为空，graph_score=0。
    失败不落盘不标记（续传自动重试）。"""
    try:
        # 无图谱：多跳路径为空，知识图谱信息为空，graph 分数为 0
        multi_hop_path = []
        graph_score = 0.0
        knowledge_graph = ""

        llm_score, llm_reason = await _llm_verify_relation_with_graph_cot(
            arm['llm'],
            sf['entity_a'],
            sf['entity_b'],
            multi_hop_path,
            context['query_document'],
            knowledge_graph,   # 空：prompt 中知识图谱部分不参与
            graph_score,       # 0：无图谱置信度
            context['entity_a_standard'],
            context['entity_b_standard'],
            None,              # rag 参数（验证函数体内未使用）
            sf['entity_a_type'],
            sf['entity_b_type']
        )

        # 与 enhanced_query_entity_relation 的最终判定一致：LLM 分数即最终分数
        final_score = llm_score
        llm_has_relation = final_score > 0.5
        llm_confidence = final_score
        reason = llm_reason
        is_correct = (llm_has_relation == sf['true_label'])

        print(f"  [{arm['name']}] 判断: {'YES' if llm_has_relation else 'NO'} | "
              f"金标: {'YES' if sf['true_label'] else 'NO'} | "
              f"{'OK' if is_correct else 'X'} | llm={llm_score:.3f}")

        result_dict = {
            'doc_id': sf['doc_id'],
            'entity_a': sf['entity_a'],
            'entity_b': sf['entity_b'],
            'entity_a_type': sf['entity_a_type'],
            'entity_b_type': sf['entity_b_type'],
            'true_label': sf['true_label'],
            'relation_type': sf['relation_type'],
            'llm_has_relation': llm_has_relation,
            'llm_confidence': llm_confidence,
            'llm_score': llm_score,
            'graph_score': graph_score,
            'is_correct': is_correct,
            'reason': reason,
            'error': None
        }

        csv_writer.append_result(result_dict)
        tested_manager.mark_tested(sf['doc_id'], sf['entity_a'], sf['entity_b'])
        return result_dict

    except Exception as e:
        print(f"  [{arm['name']}] 测试失败（未记录，续传时将重试）: {type(e).__name__}: {e}")
        return {'error': str(e)}


async def test_single_sample(
    sample: dict,
    arms: List[dict],
    managers: Dict[str, TestedSamplesManager],
    writers: Dict[str, CSVResultWriter],
    current: int,
    total: int
) -> dict:
    """单样本双臂测试：构建一次检索上下文，所有未完成臂在同一上下文上验证。
    与 test_gda.py 的唯一差异：跳过临时图谱构建步骤。"""

    sf = {
        'doc_id': sample['doc_id'],
        'entity_a': sample['entity1_text'],
        'entity_b': sample['entity2_text'],
        'entity_a_type': map_gda_type(sample['entity1_type']),
        'entity_b_type': map_gda_type(sample['entity2_type']),
        'text': sample['text'],
        'true_label': sample['true_label'],
        'relation_type': sample['relation_type']
    }

    pending = [
        arm for arm in arms
        if not managers[arm['name']].is_tested(sf['doc_id'], sf['entity_a'], sf['entity_b'])
    ]
    if not pending:
        return {'skipped': True, 'ran': 0, 'correct': 0, 'errors': 0}

    print(f"\n{'='*60}")
    print(f"[样本 {current}/{total}] DocID: {sf['doc_id']} | 待跑臂: {[a['name'] for a in pending]}")
    print(f"实体A: {sf['entity_a']} ({sf['entity_a_type']})")
    print(f"实体B: {sf['entity_b']} ({sf['entity_b_type']})")
    print(f"金标准: {'YES' if sf['true_label'] else 'NO'} ({sf['relation_type']})")
    print(f"{'='*60}")

    # 步骤1: 检索上下文（两臂共享；与 test_gda.py 完全一致）
    try:
        context = await prepare_query_context(
            sf['entity_a'], sf['entity_b'],
            abstract=sf['text'],
            enable_rerank=True
        )
    except Exception as e:
        print(f"[错误] 检索上下文构建失败（本样本各臂均未记录）: {type(e).__name__}: {e}")
        return {'skipped': False, 'ran': 0, 'correct': 0, 'errors': len(pending)}

    # 无步骤2：跳过临时图谱构建（本版本不建图，图谱不参与判断）

    # 步骤3: 各臂在同一检索上下文上依次验证
    ran = correct = errors = 0
    for arm in pending:
        result = await _run_arm(sf, context, arm, managers[arm['name']], writers[arm['name']])
        ran += 1
        if result.get('error'):
            errors += 1
        elif result.get('is_correct'):
            correct += 1

    return {'skipped': False, 'ran': ran, 'correct': correct, 'errors': errors}


# ============== 主测试函数 ==============

async def run_test(n_samples: int | None = None):
    """运行无图谱简化测试（双臂: gda约束 vs base；样本集与 test_gda.py 结果一一对应）"""

    print("="*80)
    print("GDA 关系验证测试程序（无图谱简化版，双臂: gda约束 vs base，样本与 test_gda.py 一一对应）")
    print("="*80)

    # 两臂各自的CSV写入器与已测试管理器
    writers: Dict[str, CSVResultWriter] = {}
    managers: Dict[str, TestedSamplesManager] = {}
    for arm in ARMS:
        path = CACHE_DIR / f"gda_simple_test_results_{arm['name']}.csv"
        writers[arm['name']] = CSVResultWriter(path)
        managers[arm['name']] = TestedSamplesManager(path)
        managers[arm['name']].load_tested()

    # 读取所有样本
    print(f"\n[步骤1] 读取测试集: {GDA_JSON_PATH}")
    all_samples = read_gda_json(GDA_JSON_PATH)
    print(f"  测试集总样本数: {len(all_samples)}")

    # 样本选择：与 test_gda.py 定稿结果一一对应（不做随机抽样）
    print(f"\n[步骤2] 加载参照样本集合: {REFERENCE_RESULTS_CSV}")
    ref_doc_ids = load_reference_doc_ids(REFERENCE_RESULTS_CSV)
    if not ref_doc_ids:
        print("[错误] 参照结果为空，无法确定样本集合")
        return

    selected_samples = [s for s in all_samples if s['doc_id'] in ref_doc_ids]
    found_ids = {s['doc_id'] for s in selected_samples}
    missing_ids = ref_doc_ids - found_ids
    print(f"  参照样本: {len(ref_doc_ids)} 个 | JSON中匹配到: {len(selected_samples)} 个")
    if missing_ids:
        print(f"  [警告] {len(missing_ids)} 个doc_id在JSON中未找到: {sorted(missing_ids)[:5]}")

    pos_count = sum(1 for s in selected_samples if s['true_label'])
    neg_count = len(selected_samples) - pos_count
    print(f"  样本构成: 正样本 {pos_count}, 负样本 {neg_count}")

    if n_samples is not None:
        selected_samples = selected_samples[:n_samples]
        print(f"  [调试] --n 截断为前 {n_samples} 个")

    if not selected_samples:
        print("\n[完成] 没有需要测试的样本")
        for arm in ARMS:
            stats = writers[arm['name']].get_stats()
            print(f"  [{arm['name']}] 已测试: {stats['total']}, 正确: {stats['correct']}, 准确率: {stats['accuracy']:.2%}")
        return

    # 测试循环
    print(f"\n[步骤3] 开始双臂测试（无图谱：跳过建图，仅检索上下文 + LLM验证）...")
    total_ran = total_correct = total_errors = 0
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5  # 整样本连续失败达到该数即中止（疑似断网/API异常）
    run_start = time.time()

    for i, sample in enumerate(selected_samples):
        stats = await test_single_sample(
            sample=sample,
            arms=ARMS,
            managers=managers,
            writers=writers,
            current=i + 1,
            total=len(selected_samples)
        )

        if stats['skipped']:
            continue

        total_ran += stats['ran']
        total_correct += stats['correct']
        total_errors += stats['errors']

        # 整样本全臂失败才累计连续错误（部分臂成功说明网络正常）
        if stats['errors'] == stats['ran'] and stats['ran'] > 0:
            consecutive_errors += stats['errors']
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"\n[中止] 连续 {MAX_CONSECUTIVE_ERRORS} 次验证全部失败，疑似网络/API异常，主动停止。")
                print(f"[中止] 已完成记录在各臂CSV；失败样本未标记已测试，网络恢复后重新运行即可断点续传。")
                break
        else:
            consecutive_errors = 0

        # 每 CACHE_INTERVAL 个样本显示一次进度
        if (i + 1) % CACHE_INTERVAL == 0 or (i + 1) == len(selected_samples):
            print(f"\n{'='*60}")
            print(f"[进度] 样本 {i+1}/{len(selected_samples)} | 本次验证 {total_ran} 次（正确 {total_correct}，失败 {total_errors}）")
            for arm in ARMS:
                st = writers[arm['name']].get_stats()
                print(f"  [{arm['name']}] 累计: {st['correct']}/{st['total']} ({st['accuracy']:.2%})")
            elapsed = time.time() - run_start
            done = i + 1
            print(f"  平均 {elapsed/done:.0f}s/样本 | 预计剩余 {(len(selected_samples)-done)*elapsed/done/60:.0f} 分钟")
            print(f"{'='*60}")

    # 最终统计
    print(f"\n{'='*80}")
    print(f"[测试完成] 最终统计")
    print(f"{'='*80}")
    print(f"  本次处理样本: {len(selected_samples)} 个 | 本次验证 {total_ran} 次")
    for arm in ARMS:
        st = writers[arm['name']].get_stats()
        print(f"\n  [{arm['name']}] ({CACHE_DIR / ('gda_simple_test_results_' + arm['name'] + '.csv')})")
        print(f"    累计: {st['total']} 条 | 正确: {st['correct']} | 准确率: {st['accuracy']:.2%}")


# ============== 主入口 ==============

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GDA 关系验证测试（无图谱简化版，双臂: gda约束 vs base，样本与 test_gda.py 一一对应）")
    parser.add_argument('--n', type=int, default=None, help='只测试参照样本集的前N个（调试用；默认全部）')
    args = parser.parse_args()

    print(f"\n[配置]")
    print(f"  样本集合: 与 {REFERENCE_RESULTS_CSV.name} 一一对应（{len(load_reference_doc_ids(REFERENCE_RESULTS_CSV))} 个）")
    print(f"  调试截断: {('前 ' + str(args.n) + ' 个') if args.n else '无（全部参照样本）'}")
    print(f"  双臂: {[a['name'] for a in ARMS]} | 无图谱（跳过建图，图谱不参与判断）")
    for arm in ARMS:
        print(f"  [{arm['name']}] 结果: {CACHE_DIR / ('gda_simple_test_results_' + arm['name'] + '.csv')}")

    if not QWEN_API_KEY:
        print("ERROR: DASHSCOPE_API_KEY not found")
        sys.exit(1)

    asyncio.run(run_test(n_samples=args.n))
