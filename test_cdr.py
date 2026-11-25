"""
CDR 测试程序（双臂同图设计）

功能：
1. 读取 CDR JSON 格式的测试集，分层随机抽样（正负比例 1:2）
2. 双臂对照：cdr臂（注入CDR域约束） vs base臂（主程序原生prompt）
3. 每个样本只构建一次临时图谱，两臂的最终验证调用在同一个图谱上进行
   —— 臂间唯一差异即验证 prompt，保证严格配对
4. 结果分臂写入 cdr_test_results_cdr.csv / cdr_test_results_base.csv
5. 断点续传按（样本, 臂）粒度跳过已完成记录
6. 网络自保护：失败样本不落盘不标记（续传自动重试），连续失败主动中止

运行方式: python test_cdr.py [--n 500]
"""

import asyncio
import csv
import json
import random
import shutil
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

# 导入主程序模块
from lightrag_core import (
    enhanced_query_entity_relation,
    qwen_llm_func,
    prepare_query_context,
    build_minimal_viable_temp_graph,
    openai_embed,
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_EMBEDDING_MODEL,
    TEMP_DIR_BASE
)


# ============== 配置区域（可直接修改） ==============

# 本次测试的样本数量
# None = 测试所有未测试的样本
# 数字 = 测试指定数量的样本（分层随机抽样，正负比例1:2）
N_SAMPLES = 500  # 目标规模：500个样本（约167正+333负）

# 随机种子（None 表示每次随机，设置数字可重现结果）
RANDOM_SEED = 4200

# CDR 数据文件路径
CDR_JSON_PATH = os.getenv("CDR_JSON_PATH", str(Path(__file__).resolve().parent / "data/CDR/test_new.json"))

# 缓存和结果目录
CACHE_DIR = Path(os.getenv("GRACE_RESULTS_DIR", str(Path(__file__).resolve().parent / "results")))
CACHE_DIR.mkdir(exist_ok=True)

RESULT_CSV = CACHE_DIR / "cdr_test_results.csv"

# 每 N 个样本显示一次进度
CACHE_INTERVAL = 10

# ============== CDR 域约束（仅注入最终验证调用，不影响图谱构建） ==============

CDR_CONSTRAINT = """Entity A is ALWAYS a chemical/drug; Entity B is ALWAYS a disease.
A relation ("yes") requires the abstract to explicitly link A and B
(direction never matters): etiologic (A causes / induces / exacerbates
B, including adverse reactions), biomarker (A correlates with B), or
therapeutic (A is asserted to treat B).
Assertions across sentence boundaries count. Animal and in vitro
studies count as evidence.
If the abstract asserts no such A-B association, answer "no".
Apply these rules in Step 0, BEFORE the standard checks"""


async def cdr_llm_func(prompt, system_prompt=None, **kwargs):
    """CDR域约束包装器：仅拦截最终关系验证调用（system_prompt含step-by-step标记），
    在用户prompt前部注入CDR判定口径；LightRAG三元组抽取等其他调用原样透传。"""
    if system_prompt and 'MUST follow this EXACT step-by-step format' in system_prompt:
        prompt = CDR_CONSTRAINT + '\n\n' + prompt
    return await qwen_llm_func(prompt, system_prompt=system_prompt, **kwargs)


# ============== 双臂配置 ==============
# 双臂同图：每个样本只构建一次临时图谱，两臂的最终验证在同一图谱上进行，
# 臂间唯一差异即验证 prompt（cdr = 注入CDR域约束；base = 主程序原生prompt）
ARMS = [
    {'name': 'cdr', 'llm': cdr_llm_func},
    {'name': 'base', 'llm': qwen_llm_func},
]


def cleanup_temp_graph_dirs():
    """删除临时图谱落盘目录（build_minimal_viable_temp_graph 用 mkdtemp 创建，
    名称前缀 temp_graph_）。本脚本为串行单进程，调用时刻不存在正在使用的目录，
    顺带清掉历史残留。"""
    try:
        base = Path(TEMP_DIR_BASE)
        if not base.exists():
            return
        for d in base.iterdir():
            if d.is_dir() and d.name.startswith('temp_graph_'):
                shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        print(f"[缓存清理] 失败（不影响主流程）: {e}")


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
                # 使用 (doc_id, entity_a, entity_b) 作为唯一标识
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
    """CSV结果写入器（支持追加）"""

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


# ============== CDR JSON 读取与抽样 ==============

def read_cdr_json(json_path: str) -> List[dict]:
    """读取 CDR JSON 文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for idx, item in enumerate(data):
        # CDR 格式: h (head/chemical), t (tail/disease), relation (1=正, 0=负)
        head_entity = item.get('h', {})
        tail_entity = item.get('t', {})

        samples.append({
            'doc_id': f"CDR_{idx}",
            'entity1_id': head_entity.get('id', ''),
            'entity2_id': tail_entity.get('id', ''),
            'entity1_text': head_entity.get('name', ''),
            'entity2_text': tail_entity.get('name', ''),
            'entity1_type': head_entity.get('type', ''),
            'entity2_type': tail_entity.get('type', ''),
            'text': item.get('token', ''),
            'true_label': item.get('relation', 0) == 1,
            'relation_type': 'Chemical-DiseaseRelation' if item.get('relation', 0) == 1 else ''
        })

    return samples


def stratified_random_sample_1to2(
    samples: List[dict],
    n_samples: int,
    tested_managers: List[TestedSamplesManager],
    random_seed: int = None
) -> List[dict]:
    """分层随机抽样（正负样本比例 1:2）"""

    if random_seed is not None:
        random.seed(random_seed)

    # 分离正负样本
    positive_samples = [s for s in samples if s['true_label']]
    negative_samples = [s for s in samples if not s['true_label']]

    # 过滤所有臂均已完成（任一臂未完成即需保留，补跑该臂）
    def filter_untested(sample_list):
        return [
            s for s in sample_list
            if not all(
                m.is_tested(s['doc_id'], s['entity1_text'], s['entity2_text'])
                for m in tested_managers
            )
        ]

    positive_untested = filter_untested(positive_samples)
    negative_untested = filter_untested(negative_samples)

    print(f"[样本统计]")
    print(f"  正样本总数: {len(positive_samples)}, 未测试: {len(positive_untested)}")
    print(f"  负样本总数: {len(negative_samples)}, 未测试: {len(negative_untested)}")

    # 按 1:2 比例计算正负样本数量
    # n_samples = n_pos + n_neg, 且 n_pos : n_neg = 1 : 2
    # 所以 n_pos = n_samples / 3, n_neg = 2 * n_samples / 3
    n_pos = max(1, min(int(n_samples / 3), len(positive_untested)))
    n_neg = min(n_samples - n_pos, len(negative_untested))

    # 如果负样本不足，调整正样本数量
    if n_neg < 2 * n_pos:
        n_pos = min(int(n_neg / 2), len(positive_untested))

    print(f"[抽样设置] 正负比例 1:2")
    print(f"  目标正样本: {n_pos}")
    print(f"  目标负样本: {n_neg}")
    print(f"  总计: {n_pos + n_neg}")

    # 随机抽样
    selected = []
    if n_pos > 0:
        selected.extend(random.sample(positive_untested, n_pos))
    if n_neg > 0:
        selected.extend(random.sample(negative_untested, n_neg))

    # 打乱顺序
    random.shuffle(selected)

    actual_pos = sum(1 for s in selected if s['true_label'])
    actual_neg = len(selected) - actual_pos
    print(f"[抽样结果]")
    print(f"  实际正样本: {actual_pos}")
    print(f"  实际负样本: {actual_neg}")
    print(f"  实际比例: 1:{actual_neg/actual_pos if actual_pos > 0 else 'N/A'}")
    print(f"  总计: {len(selected)}")

    return selected


# ============== 类型映射 ==============

def map_cdr_type(cdr_type: str) -> str:
    """映射 CDR 类型到主程序类型"""
    type_mapping = {
        'Chemical': 'ChemicalOrDrug',
        'Disease': 'Disease'
    }
    return type_mapping.get(cdr_type, cdr_type)


# ============== 测试函数 ==============

async def _run_arm(
    sf: dict,
    temp_graph,
    arm: dict,
    tested_manager: TestedSamplesManager,
    csv_writer: CSVResultWriter,
) -> dict:
    """单臂验证：在给定临时图谱上跑最终验证调用并落盘。失败不落盘不标记（续传自动重试）"""
    try:
        result = await enhanced_query_entity_relation(
            rag=None,  # 该参数不参与主流程
            entity_a=sf['entity_a'],
            entity_b=sf['entity_b'],
            llm_func=arm['llm'],
            abstract=sf['text'],
            enable_rerank=True,
            entity_a_type=sf['entity_a_type'],
            entity_b_type=sf['entity_b_type'],
            prebuilt_graph=temp_graph
        )

        llm_has_relation = result.get('has_relation', False)
        llm_confidence = result.get('confidence', 0.0)
        llm_score = result.get('llm_score', 0.0)
        graph_score = result.get('graph_score', 0.0)
        reason = result.get('reason', '')
        is_correct = (llm_has_relation == sf['true_label'])

        print(f"  [{arm['name']}] 判断: {'YES' if llm_has_relation else 'NO'} | "
              f"金标: {'YES' if sf['true_label'] else 'NO'} | "
              f"{'OK' if is_correct else 'X'} | graph={graph_score:.3f} llm={llm_score:.3f}")

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
    """单样本双臂测试：构建一次临时图谱，所有未完成臂在同一图谱上验证"""

    sf = {
        'doc_id': sample['doc_id'],
        'entity_a': sample['entity1_text'],
        'entity_b': sample['entity2_text'],
        'entity_a_type': map_cdr_type(sample['entity1_type']),
        'entity_b_type': map_cdr_type(sample['entity2_type']),
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

    # 步骤1: 检索上下文（两臂共享）
    try:
        context = await prepare_query_context(
            sf['entity_a'], sf['entity_b'],
            abstract=sf['text'],
            enable_rerank=True
        )
    except Exception as e:
        print(f"[错误] 检索上下文构建失败（本样本各臂均未记录）: {type(e).__name__}: {e}")
        return {'skipped': False, 'ran': 0, 'correct': 0, 'errors': len(pending)}

    # 步骤2: 构建临时图谱（只建一次，两臂共用）
    embedding_config = {
        'func': openai_embed.func,
        'model': QWEN_EMBEDDING_MODEL,
        'api_key': QWEN_API_KEY,
        'base_url': QWEN_BASE_URL
    }
    try:
        temp_graph = await build_minimal_viable_temp_graph(
            pubmed_data=context['pubmed_data'],
            llm_func=qwen_llm_func,
            embedding_func_config=embedding_config,
            query_entity_a=context['entity_a_standard'] if context['entity_a_standard'] else sf['entity_a'],
            query_entity_b=context['entity_b_standard'] if context['entity_b_standard'] else sf['entity_b'],
            entity_a_standard=context['entity_a_standard'],
            entity_a_cui=None,
            entity_b_standard=context['entity_b_standard'],
            entity_b_cui=None
        )
    except Exception as e:
        print(f"[错误] 临时图谱构建失败（本样本各臂均未记录）: {type(e).__name__}: {e}")
        return {'skipped': False, 'ran': 0, 'correct': 0, 'errors': len(pending)}

    # 步骤3: 各臂在同一图谱上依次验证
    ran = correct = errors = 0
    try:
        for arm in pending:
            result = await _run_arm(sf, temp_graph, arm, managers[arm['name']], writers[arm['name']])
            ran += 1
            if result.get('error'):
                errors += 1
            elif result.get('is_correct'):
                correct += 1
    finally:
        cleanup_temp_graph_dirs()

    return {'skipped': False, 'ran': ran, 'correct': correct, 'errors': errors}


# ============== 主测试函数 ==============

async def run_test(n_samples: int | None = None, random_seed: int = None):
    """运行双臂同图测试"""

    print("="*80)
    print("CDR 关系验证测试程序（双臂同图: cdr约束 vs base，正负比例 1:2）")
    print("="*80)

    # 两臂各自的CSV写入器与已测试管理器
    writers: Dict[str, CSVResultWriter] = {}
    managers: Dict[str, TestedSamplesManager] = {}
    for arm in ARMS:
        path = CACHE_DIR / f"cdr_test_results_{arm['name']}.csv"
        writers[arm['name']] = CSVResultWriter(path)
        managers[arm['name']] = TestedSamplesManager(path)
        managers[arm['name']].load_tested()

    # 读取所有样本
    print(f"\n[步骤1] 读取测试集: {CDR_JSON_PATH}")
    all_samples = read_cdr_json(CDR_JSON_PATH)
    total_available = len(all_samples)
    print(f"  总样本数: {total_available}")

    # 统计正负样本
    pos_count = sum(1 for s in all_samples if s['true_label'])
    neg_count = total_available - pos_count
    print(f"  正样本: {pos_count}, 负样本: {neg_count}")
    print(f"  原始比例: 1:{neg_count/pos_count if pos_count > 0 else 'N/A'}")

    # 如果 n_samples 为 None，测试所有未完成（任一臂未跑）的样本
    if n_samples is None:
        untested_count = sum(
            1 for s in all_samples
            if not all(
                m.is_tested(s['doc_id'], s['entity1_text'], s['entity2_text'])
                for m in managers.values()
            )
        )
        n_samples = untested_count
        print(f"  [自动设置] 将处理所有 {n_samples} 个未完成样本")

    # 分层随机抽样 (1:2 比例)
    print(f"\n[步骤2] 分层随机抽样 (目标: {n_samples} 个样本, 正负比例 1:2)")
    selected_samples = stratified_random_sample_1to2(
        samples=all_samples,
        n_samples=n_samples,
        tested_managers=list(managers.values()),
        random_seed=random_seed
    )

    if not selected_samples:
        print("\n[完成] 没有新的样本需要测试（两臂均已完成）")
        for arm in ARMS:
            stats = writers[arm['name']].get_stats()
            print(f"  [{arm['name']}] 已测试: {stats['total']}, 正确: {stats['correct']}, 准确率: {stats['accuracy']:.2%}")
        return

    # 测试循环
    print(f"\n[步骤3] 开始双臂测试...")
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
        print(f"\n  [{arm['name']}] ({CACHE_DIR / ('cdr_test_results_' + arm['name'] + '.csv')})")
        print(f"    累计: {st['total']} 条 | 正确: {st['correct']} | 准确率: {st['accuracy']:.2%}")


# ============== 主入口 ==============

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CDR 关系验证测试（GRACE 双臂同图: cdr约束 vs base）")
    parser.add_argument('--n', type=int, default=None, help='测试样本数（默认=N_SAMPLES）')
    args = parser.parse_args()

    n_samples = args.n if args.n is not None else N_SAMPLES

    print(f"\n[配置]")
    print(f"  测试样本数: {n_samples} (正负比例 1:2)")
    print(f"  随机种子: {RANDOM_SEED if RANDOM_SEED else '未设置（每次随机）'}")
    print(f"  双臂: {[a['name'] for a in ARMS]} | 同图构建（每样本一次）")
    for arm in ARMS:
        print(f"  结果文件[{arm['name']}]: {CACHE_DIR / ('cdr_test_results_' + arm['name'] + '.csv')}")

    asyncio.run(run_test(n_samples=n_samples, random_seed=RANDOM_SEED))
