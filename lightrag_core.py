"""
LightRAG Core Module

导出的函数:
- enhanced_query_entity_relation()
- initialize_rag()
- qwen_llm_func()
- QWEN_API_KEY (常量)
"""

import os
import sys

# 限制 BLAS 线程数以规避其内存分配问题（必须在 import numpy 之前设置）
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import asyncio
import json
import math
import re
import shutil
import tempfile
import uuid
import hashlib
import logging
from functools import partial
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import networkx as nx

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# PromptManager for using shared prompts
try:
    from lightrag.prompts import get_prompt_manager
except ImportError:
    # 如果导入失败，使用fallback函数
    print("[警告] 无法导入PromptManager，使用默认prompt")
    def get_prompt_manager():
        class FakeManager:
            def get_prompt(self, name, variant='basic'):
                # 返回默认的fallback prompt
                return "Verify if {entity_a} and {entity_b} have a meaningful medical relationship."
        return FakeManager()

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.rerank import create_aliyun_rerank_func
from dotenv import load_dotenv

# BioRED RAG路径
biored_rag_path = str(Path(__file__).resolve().parent)
if biored_rag_path not in sys.path:
    sys.path.insert(0, biored_rag_path)

# 导入动态RAG模块
try:
    from syRACT_dynamic_rag_with_omop import EnhancedDynamicRAG
    DYNAMIC_RAG_AVAILABLE = True
except ImportError as e:
    raise ImportError(f"动态RAG模块导入失败，请确认 syRACT_dynamic_rag_with_omop.py 与本文件位于同一目录: {e}")

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)


# ============== 配置常量 ==============

# 医学实体类型（用于实体识别和关系抽取）
MEDICAL_ENTITY_TYPES = [
    "Disease",
    "ChemicalOrDrug",
    "GeneOrProtein",
    "Symptom"
]

# 临时图谱缓存目录
TEMP_DIR_BASE = Path(os.getenv("GRACE_TEMP_DIR", str(Path(__file__).resolve().parent / "temp_workspace")))
TEMP_DIR_BASE.mkdir(exist_ok=True)

# LightRAG工作目录（主实例存储不参与验证流程，仅承载库初始化文件，挂在临时工作区下）
WORKING_DIR = str(TEMP_DIR_BASE / "rag_storage")

# Qwen API Configuration
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")
QWEN_EMBEDDING_DIM = int(os.getenv("QWEN_EMBEDDING_DIM", "1024"))

# ============== LLM函数 ==============

async def qwen_llm_func(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    temperature=0.0,
    max_tokens=2000,
    top_p=0.9,
    **kwargs
):
    """Qwen LLM function using OpenAI-compatible API"""
    if history_messages is None:
        history_messages = []

    return await openai_complete_if_cache(
        QWEN_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        **kwargs
    )


# ============== LightRAG初始化 ==============

async def initialize_rag(working_dir: str = None, enable_rerank: bool = True):
    """Initialize LightRAG with Qwen configuration and medical entity types

    Args:
        working_dir: Optional working directory path, defaults to global WORKING_DIR
        enable_rerank: 是否启用Rerank功能（默认启用）
    """
    actual_working_dir = working_dir or WORKING_DIR

    # 创建Rerank函数
    rerank_func = None
    if enable_rerank:
        print("[Rerank] 初始化通义千问 Rerank API (gte-rerank-v2)...")
        rerank_func = create_aliyun_rerank_func(
            model="gte-rerank-v2",
            top_n=10,
            api_key=QWEN_API_KEY  # 使用已有的 Qwen API Key
        )
        print("[Rerank] Rerank函数已创建（使用通义千问 API）")

    rag = LightRAG(
        working_dir=actual_working_dir,
        llm_model_func=qwen_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=QWEN_EMBEDDING_DIM,
            max_token_size=8192,
            func=partial(
                openai_embed.func,
                model=QWEN_EMBEDDING_MODEL,
                api_key=QWEN_API_KEY,
                base_url=QWEN_BASE_URL,
            ),
        ),
        addon_params={
            "entity_types": MEDICAL_ENTITY_TYPES
        },
        rerank_model_func=rerank_func,  # 添加rerank函数
        min_rerank_score=0.4  # rerank分数阈值，低于此分数的chunks被过滤
    )
    await rag.initialize_storages()
    return rag

# ============== 辅助函数 ==============

async def _find_similar_entity_in_graph(query_entity: str, rag_instance, threshold: float = 0.3) -> str:
    """在图谱中查找与查询实体相似的节点（支持大小写不敏感匹配）"""
    try:
        from difflib import SequenceMatcher

        if not hasattr(rag_instance, 'chunk_entity_relation_graph'):
            return ""

        graph = await rag_instance.chunk_entity_relation_graph._get_graph()
        all_nodes = list(graph.nodes())

        best_match = ""
        best_similarity = 0.0
        query_lower = query_entity.lower().strip()

        # 先尝试大小写不敏感的精确匹配
        for node in all_nodes:
            node_lower = node.lower().strip()
            if node_lower == query_lower:
                return node  # 返回实际节点名称（保持原始大小写）

        # 如果没有精确匹配，再尝试相似度匹配
        for node in all_nodes:
            node_lower = node.lower().strip()

            # 检查包含关系
            if query_lower in node_lower and len(query_lower) > 3:
                similarity = len(query_lower) / len(node_lower)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = node

            # 序列相似度
            similarity = SequenceMatcher(None, query_lower, node_lower).ratio()
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match = node

        if best_match:
            print(f"  [模糊匹配] 找到相似节点: '{query_entity}' -> '{best_match}' (相似度: {best_similarity:.2f})")

        return best_match

    except Exception as e:
        print(f"  [模糊匹配] 查找失败: {e}")
        return ""


async def _ensure_graph_node_name(entity_name: str, rag_instance) -> str:
    """确保使用图谱中实际存在的节点名

    如果图谱中存在该实体（大小写不敏感匹配），返回图谱中的实际节点名。
    否则返回原名称。

    Args:
        entity_name: 查询实体名称
        rag_instance: LightRAG实例

    Returns:
        图谱中的实际节点名，如果不存在则返回原名称
    """
    if not entity_name or not hasattr(rag_instance, 'chunk_entity_relation_graph'):
        return entity_name

    try:
        graph = await rag_instance.chunk_entity_relation_graph._get_graph()
        all_nodes = list(graph.nodes())

        entity_lower = entity_name.lower().strip()

        # 先尝试大小写不敏感的精确匹配
        node_lower_to_original = {n.lower().strip(): n for n in all_nodes}

        if entity_lower in node_lower_to_original:
            actual_name = node_lower_to_original[entity_lower]
            if actual_name != entity_name:
                print(f"  [节点名修正] '{entity_name}' -> '{actual_name}' (图谱中的实际节点名)")
            return actual_name

        # 如果没有精确匹配，尝试相似度匹配（调用已有函数）
        similar_name = await _find_similar_entity_in_graph(entity_name, rag_instance)
        if similar_name:
            return similar_name

        # 都没找到，返回原名称
        return entity_name

    except Exception as e:
        print(f"  [节点名修正] 查找失败: {e}")
        return entity_name


async def _find_multi_hop_path_in_graph(entity_a: str, entity_b: str, rag_instance, max_hops: int = 3) -> list:
    """在图谱中查找两个实体之间的多跳路径（支持大小写不敏感匹配）"""
    try:
        if not hasattr(rag_instance, 'chunk_entity_relation_graph'):
            return []

        graph = await rag_instance.chunk_entity_relation_graph._get_graph()
        all_nodes = list(graph.nodes())
        node_lower_to_original = {n.lower().strip(): n for n in all_nodes}

        entity_a_lower = entity_a.lower().strip()
        entity_b_lower = entity_b.lower().strip()

        if entity_a_lower not in node_lower_to_original:
            print(f"  [多跳路径] 警告: 节点 '{entity_a}' 不在图谱中")
            return []
        if entity_b_lower not in node_lower_to_original:
            print(f"  [多跳路径] 警告: 节点 '{entity_b}' 不在图谱中")
            return []

        # 使用实际节点名进行查询
        actual_a = node_lower_to_original[entity_a_lower]
        actual_b = node_lower_to_original[entity_b_lower]

        try:
            path = nx.shortest_path(graph, source=actual_a, target=actual_b)
            path_length = len(path) - 1

            if path_length <= max_hops:
                return path
            else:
                return []

        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            return []

    except Exception as e:
        return []


def _extract_multi_hop_path(query_result: str) -> list:
    """从查询结果中提取多跳路径"""
    if not query_result:
        return []

    invalid_indicators = [
        'no information', 'no context', 'cannot provide',
        'unable to answer', 'no data available', '[no-context]'
    ]
    query_lower = query_result.lower()
    if any(indicator in query_lower for indicator in invalid_indicators):
        return []

    path = []

    # 尝试提取路径模式：A -> C -> D -> E -> B
    patterns = [
        r'path[:\s]*([A-Za-z\s]+(?:\s*->\s*[A-Za-z\s]+)+)',
        r'([A-Za-z\s]+)(?:\s*->\s*[A-Za-z\s]+)(?:\s*->\s*[A-Za-z\s]+)+',
        r'(?:relation|relationship)[s]?:\s*([^\n]+)',
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, query_result, re.IGNORECASE)
        for match in matches:
            path_text = match.group(1) if match else match.group(0)
            entities = [e.strip() for e in re.split(r'->|,|\sand\s', path_text)]
            entities = [e for e in entities if e and len(e) > 1]
            if entities:
                path = entities
                break

        if path:
            break

    return path


# ========== 优先级2：路径语义验证常量和函数 ==========

# 定义七种目标关系类型
TARGET_RELATION_TYPES = {
    'treats',           # 治疗
    'causes',           # 导致
    'pharmacological',  # 药理作用
    'receptor',         # 受体-配体
    'enzyme',           # 酶-底物
    'drug-disease',     # 药物-疾病关联
    'gene-disease',     # 基因-疾病关联
    'pharmacological effect',  # 药理作用
    'receptor-ligand',  # 受体-配体
    'enzyme-substrate',  # 酶-底物
    'gene-disease association'  # 基因-疾病关联
}

# 定义无效关系模式（这些关系不应该出现在医学关系路径中）
# 与验证 prompt Step 2 的 Exclusion 词表保持一致（hierarchy, diagnostic, indirect, shared pathway）
INVALID_RELATION_PATTERNS = [
    'expresses', 'expression', 'expression pattern',
    'found in', 'located in', 'contains',
    'developmental switch', 'stage-specific',
    'timing coincides', 'present in', 'absent in',
    'hierarchy', 'diagnostic', 'indirect', 'shared pathway'
]

# 有效的边关系关键词（用于正向验证）
VALID_RELATION_KEYWORDS = {
    'treats': ['treats', 'therapeutic', 'therapy', 'remedy', 'effective against', 'treatment'],
    'causes': ['causes', 'induces', 'leads to', 'results in', 'triggers', 'caused by'],
    'pharmacological': ['relieves', 'reduces', 'inhibits', 'activates', 'improves', 'prevents', 'pharmacological'],
    'receptor': ['binds', 'receptor', 'ligand', 'binding', 'affinity'],
    'enzyme': ['metabolizes', 'catalyzes', 'substrate', 'enzymatic', 'breakdown', 'enzyme'],
    'drug-disease': ['correlation', 'risk', 'associated', 'drug-disease'],
    'gene-disease': ['associated', 'implicated', 'linked', 'mutation', 'variant', 'dysregulation', 'overexpression', 'loss', 'gene-disease']
}


async def _validate_path_semantics(multi_hop_path: list, temp_graph) -> tuple:
    """验证路径的语义合理性并收集边数据

    检查：
    1. 路径上的关系类型是否属于目标类型
    2. 过滤包含无效关系模式的边
    3. 统计有效边数
    4. 收集边数据供后续使用（避免重复获取）

    Args:
        multi_hop_path: 实体路径列表
        temp_graph: LightRAG实例

    Returns:
        (is_valid, invalid_reason, valid_edge_count, total_edges, edge_data_list)
        其中 edge_data_list 是一个列表，每个元素包含边的数据和置信度信息
    """
    if not multi_hop_path or len(multi_hop_path) < 2:
        return False, "路径太短", 0, 0, []

    graph_instance = temp_graph.chunk_entity_relation_graph
    valid_edge_count = 0
    invalid_reasons = []
    edge_data_list = []

    # 获取所有节点（用于大小写不敏感匹配）
    try:
        all_nodes = list((await graph_instance._get_graph()).nodes())
        node_lower_to_original = {n.lower(): n for n in all_nodes}
    except Exception as e:
        print(f"  [路径验证] 警告：无法获取节点列表: {e}")
        all_nodes = []
        node_lower_to_original = {}

    # 遍历路径上的每一条边
    for i in range(len(multi_hop_path) - 1):
        source = multi_hop_path[i]
        target = multi_hop_path[i + 1]

        # 先尝试精确匹配
        try:
            edge_data = await graph_instance.get_edge(source, target)
        except Exception:
            edge_data = None

        # 如果精确匹配失败，尝试大小写不敏感匹配
        if not edge_data:
            source_lower = source.lower().strip()
            target_lower = target.lower().strip()

            # 查找实际节点名称
            actual_source = node_lower_to_original.get(source_lower)
            actual_target = node_lower_to_original.get(target_lower)

            if actual_source and actual_target:
                try:
                    edge_data = await graph_instance.get_edge(actual_source, actual_target)
                    if edge_data:
                        print(f"  [路径验证] 大小写修正: '{source}' -> '{target}' 实际为 '{actual_source}' -> '{actual_target}'")
                except Exception:
                    pass

            # 如果大小写修正仍然失败，尝试相似度匹配
            if not edge_data:
                fuzzy_source = await _find_similar_entity_in_graph(source, temp_graph)
                fuzzy_target = await _find_similar_entity_in_graph(target, temp_graph)

                if fuzzy_source and fuzzy_target:
                    try:
                        edge_data = await graph_instance.get_edge(fuzzy_source, fuzzy_target)
                        if edge_data:
                            print(f"  [路径验证] 相似度修正: '{source}' -> '{target}' 实际为 '{fuzzy_source}' -> '{fuzzy_target}'")
                    except Exception:
                        pass

        if not edge_data:
            invalid_reasons.append(f"边 '{source}' -> '{target}' 不存在")
            # 添加默认边数据（用于后续计算）
            edge_data_list.append({
                'source': source,
                'target': target,
                'edge_data': None,
                'confidence': 0.1
            })
            continue

        # 检查关系类型（通过keywords字段）
        keywords = edge_data.get('keywords', '')
        keywords_lower = keywords.lower()

        # 检查1：是否包含无效关系模式
        has_invalid_pattern = any(pattern in keywords_lower for pattern in INVALID_RELATION_PATTERNS)
        if has_invalid_pattern:
            invalid_reasons.append(f"边 '{source}' -> '{target}' 包含无效关系模式")
            # 无效边也添加到列表，但置信度设为0
            edge_data_list.append({
                'source': source,
                'target': target,
                'edge_data': edge_data,
                'confidence': 0.0
            })
            continue

        # 检查2：是否包含有效的关系关键词
        has_valid_keyword = False
        for rel_type, kw_list in VALID_RELATION_KEYWORDS.items():
            if any(kw in keywords_lower for kw in kw_list):
                has_valid_keyword = True
                break

        # 含有效关系关键词的边计数（仅用于日志展示）
        if has_valid_keyword:
            valid_edge_count += 1

        # 提取置信度（与_calculate_graph_score保持一致）
        confidence = edge_data.get('confidence_score', None)
        if confidence is None:
            confidence = float(edge_data.get('weight', 0.5))

        # 置信度截断到[0,1]：多证据合并的 confidence_score 恒在该范围内
        confidence = max(0.0, min(1.0, float(confidence)))

        edge_data_list.append({
            'source': source,
            'target': target,
            'edge_data': edge_data,
            'confidence': confidence
        })

    total_edges = len(multi_hop_path) - 1
    is_valid = len(invalid_reasons) == 0
    reason = "; ".join(invalid_reasons) if invalid_reasons else "路径语义合理"

    return is_valid, reason, valid_edge_count, total_edges, edge_data_list


# 路径长度惩罚表（λ）：节点数 -> 惩罚系数
DEFAULT_LAMBDA_TABLE = {2: 1.0, 3: 0.70, 4: 0.50, 5: 0.35}


async def _calculate_graph_score(multi_hop_path: list, temp_graph, lambda_table: dict = None) -> float:
    """基于多跳路径计算图谱分数（改进版：使用边上存储的置信度）

    Args:
        multi_hop_path: 实体路径列表，如 [entity_a, intermediate, entity_b]
        temp_graph: LightRAG实例

    Returns:
        float: 图谱分数 (0.0-1.0)
    """
    if not multi_hop_path or len(multi_hop_path) < 2:
        return 0.0

    # 检查无效指示词（保留原有逻辑）
    invalid_indicators = ['able', 'provide', 'answer', 'question', 'context', 'information', 'no-context', 'unable']
    path_str = ' '.join(str(p).lower() for p in multi_hop_path)

    if any(indicator in path_str for indicator in invalid_indicators):
        return 0.0

    # 验证路径语义合理性并获取边数据
    is_semantic_valid, semantic_reason, valid_edge_count, total_edges, edge_data_list = await _validate_path_semantics(multi_hop_path, temp_graph)

    if not is_semantic_valid:
        print(f"  [Graph分数] 路径语义验证失败: {semantic_reason}")
        return 0.0

    print(f"  [Graph分数] 路径语义验证通过，有效边数: {valid_edge_count}/{total_edges}")

    # 从边数据列表中提取置信度（避免重复获取边数据）
    edge_confidences = [edge_item['confidence'] for edge_item in edge_data_list]

    if not edge_confidences:
        return 0.0

    # 使用调和平均数代替最小置信度
    if len(edge_confidences) == 1:
        path_confidence = edge_confidences[0]
    else:
        # 调和平均数：n / sum(1/x_i)
        reciprocal_sum = sum(1.0 / (c + 1e-6) for c in edge_confidences)
        path_confidence = len(edge_confidences) / reciprocal_sum

    # 根据路径长度计算惩罚系数（λ表；超过表长的路径线性衰减）
    path_length = len(multi_hop_path)
    table = lambda_table if lambda_table is not None else DEFAULT_LAMBDA_TABLE
    if path_length in table:
        base_penalty = table[path_length]
    else:
        longest = max(table)
        base_penalty = max(0.3, table[longest] - 0.1 * (path_length - longest))

    # 最终分数
    final_score = path_confidence * base_penalty

    print(f"  [Graph分数] 边置信度: {edge_confidences}, 调和平均: {path_confidence:.3f}")
    print(f"  [Graph分数] 路径长度: {path_length}, 惩罚: {base_penalty:.3f}")
    print(f"  [Graph分数] 最终分数: {final_score:.3f}")

    return final_score


async def format_path_for_llm(multi_hop_path: list, edge_data_list: list = None) -> str:
    """
    将知识图谱多跳路径格式化为LLM可理解的详细文本

    格式包含：
    - 路径总览
    - 每一步的关系类型、置信度、关键词
    - 路径解读和可靠性评估

    Args:
        multi_hop_path: 实体路径列表，如 [entity_a, intermediate, entity_b]
        edge_data_list: 边数据列表（来自_validate_path_semantics），格式为：
                        [{'source': str, 'target': str, 'edge_data': dict, 'confidence': float}, ...]
                        如果为None，则生成简化版路径信息

    Returns:
        格式化后的路径信息文本
    """
    if not multi_hop_path or len(multi_hop_path) < 2:
        return "**知识图谱路径**: 未找到直接路径（两个实体之间无已知连接）"

    path_info = []

    # === 路径总览 ===
    path_length = len(multi_hop_path) - 1
    path_info.append(f"## 知识图谱多跳路径分析 (长度={path_length}跳)")

    # 计算整体路径可靠性
    edge_confidences = []
    valid_relations = []

    # 如果没有边数据，返回简化版
    if not edge_data_list:
        path_info.append(f"\n**路径**: {' -> '.join(multi_hop_path)}")
        path_info.append(f"\n_注：未获取到详细边数据信息_")
        return "\n".join(path_info)

    # === 详细路径信息 ===
    for i in range(len(multi_hop_path) - 1):
        source = multi_hop_path[i]
        target = multi_hop_path[i + 1]

        hop_info = f"\n### 第{i+1}步: {source} → {target}"

        # 使用传入的边数据
        edge_item = edge_data_list[i] if i < len(edge_data_list) else None
        edge_data = edge_item.get('edge_data') if edge_item else None
        confidence = edge_item.get('confidence', 0.1) if edge_item else 0.1

        if edge_data:
            # 提取边信息
            relation_type = edge_data.get('relation_type', 'unknown')
            keywords = edge_data.get('keywords', '')
            description = edge_data.get('description', '')

            # 统计
            if confidence > 0:
                edge_confidences.append(confidence)
            if relation_type != 'unknown':
                valid_relations.append(relation_type)

            # 格式化置信度等级
            if confidence >= 0.8:
                reliability = "高可靠性"
            elif confidence >= 0.6:
                reliability = "中等可靠性"
            elif confidence >= 0.4:
                reliability = "低可靠性"
            else:
                reliability = "极低可靠性"

            hop_info += f"\n- **关系类型**: {relation_type}"
            hop_info += f"\n- **置信度**: {confidence:.2f} ({reliability})"

            if keywords:
                hop_info += f"\n- **关键词**: {keywords}"

            if description:
                # 限制描述长度
                desc_short = description[:150] + "..." if len(description) > 150 else description
                hop_info += f"\n- **描述**: {desc_short}"
        else:
            hop_info += "\n- _边数据未找到_"
            edge_confidences.append(confidence)

        path_info.append(hop_info)

    # === 路径总结 ===
    path_info.append(f"\n## 路径可靠性评估")

    if edge_confidences:
        # 计算平均置信度
        avg_confidence = sum(edge_confidences) / len(edge_confidences)

        # 计算路径质量
        if avg_confidence >= 0.7:
            path_quality = "强"
        elif avg_confidence >= 0.5:
            path_quality = "中等"
        else:
            path_quality = "弱"

        path_info.append(f"\n- **平均置信度**: {avg_confidence:.2f}")
        path_info.append(f"\n- **路径质量**: {path_quality}")
        path_info.append(f"\n- **有效关系数**: {len(valid_relations)}/{len(multi_hop_path)-1}")

    # === 路径解读 ===
    path_info.append(f"\n## 路径解读")
    path_info.append(f"\n此路径显示从 **{multi_hop_path[0]}** 到 **{multi_hop_path[-1]}** 的可能连接机制：")

    if path_length == 1:
        path_info.append(f"\n- 直接关系：两个实体之间存在1步连接")
    elif path_length == 2:
        path_info.append(f"\n- 单中介关系：通过 **{multi_hop_path[1]}** 作为中介连接")
    else:
        intermediates = " → ".join(multi_hop_path[1:-1])
        path_info.append(f"\n- 多跳关系：通过多个中介实体连接")
        path_info.append(f"\n- 中介链: {intermediates}")

    # === 使用建议 ===
    path_info.append(f"\n## 对LLM判断的建议")
    path_info.append(f"\n此路径信息来自已构建的知识图谱，请：")
    path_info.append(f"\n1. **结合PubMed摘要**：图谱路径提供连接线索，摘要提供具体证据")
    path_info.append(f"\n2. **评估每步合理性**：检查每个关系是否符合医学常识")
    path_info.append(f"\n3. **考虑路径长度**：路径越长，间接性越强，需要更强的证据支持")

    return "\n".join(path_info)


def _infer_relation_type(response_lower: str) -> str:
    """从LLM响应文本推断关系类型关键词"""
    for keyword in ["causes", "causation", "treats", "treatment", "associated", "association",
                    "increases", "decreases", "leads", "induces", "prevents"]:
        if keyword in response_lower:
            return keyword
    return "unknown"


def _reconcile_score_with_answer(llm_says_yes: bool, llm_score: float) -> float:
    """LLM返回的answer与confidence矛盾时强制对齐（no→0.2, yes→0.8）"""
    if not llm_says_yes and llm_score > 0.5:
        print(f"    [WARNING] LLM返回矛盾答案: answer=no但confidence={llm_score:.2f}，调整返回分数为0.2")
        return 0.2  # 使用低分表示"无关系"
    if llm_says_yes and llm_score <= 0.5:
        print(f"    [WARNING] LLM返回矛盾答案: answer=yes但confidence={llm_score:.2f}，调整返回分数为0.8")
        return 0.8  # 使用高分表示"有关系"
    return llm_score


class InvalidLLMVerdictError(Exception):
    """LLM 回答缺少机器可校验的 answer/confidence（或数值越界[0,1]）。

    必须按失败上抛给调用方处理（重试或标记失败），禁止按 NO 计分——
    否则伪成功回答（{}、"Unable to process request." 等）会在金标准为 NO 时
    被白记为正确，污染统计。注意：本异常继承 Exception 而非 ValueError，
    不会被 JSON 回退解析的 except 分支吞掉。"""


async def _llm_verify_relation_with_graph_cot(
    llm_func,
    entity_a: str,
    entity_b: str,
    multi_hop_path: list,
    query_document: str = "",
    knowledge_graph: str = "",
    graph_score: float = 0.0,
    entity_a_standard: str = None,
    entity_b_standard: str = None,
    rag=None,
    entity_a_type: str = "",
    entity_b_type: str = ""
) -> tuple:
    """
    使用知识图谱路径 + CoT推理验证关系

    Args:
        query_document: Query Document文本（来自csv文件的abstract或实时检索的摘要）
        knowledge_graph: 知识图谱查询结果（多跳路径信息）
        entity_a_standard: 实体A的标准化名称（如果存在）
        entity_b_standard: 实体B的标准化名称（如果存在）
        entity_a_type: 实体A的类型
        entity_b_type: 实体B的类型

    Returns:
        (llm_score, llm_reason)
    """
    print(f"    [Graph+CoT] 开始LLM验证（基于知识图谱路径+Query Document）...")

    # 使用PromptManager获取模板
    manager = get_prompt_manager()

    prompt_variant = getattr(enhanced_query_entity_relation, '_prompt_variant', 'syract_enhanced_with_kg')
    base_prompt = manager.get_prompt('relation_verification', prompt_variant)

    formatted_base_prompt = base_prompt.format(
        entity_a=entity_a,
        entity_b=entity_b,
        entity_a_type=entity_a_type or 'Unknown',
        entity_b_type=entity_b_type or 'Unknown',
        abstract=query_document[:2000] if query_document else "",  # CoT风格使用
        query_document=query_document,  # SyRCT风格使用
        knowledge_graph=knowledge_graph,  # Knowledge Graph RAG增强
        graph_score=f"{graph_score:.2f}"  # Graph-LLM Fusion增强
    )
    print(f"    [Prompt] Query Document: {len(query_document)} 字符, Knowledge Graph: {len(knowledge_graph)} 字符")
    if knowledge_graph:
        print(f"    [调试] Knowledge Graph内容:\n{knowledge_graph}\n")
    if graph_score:
        print(f"    [调试] Graph Path Confidence (graph_score): {graph_score:.2f}\n")

    try:
        # 添加system prompt以满足Qwen的JSON模式要求
        system_prompt = """You must respond with a JSON object containing EXACTLY these three fields:
1. "answer": string, either "yes" or "no" (lowercase)
2. "confidence": number, a float between 0.0 and 1.0 (NOT a boolean!)
3. "reason": string, your explanation

DO NOT use these field names: relationship_exists, has_relation, result, is_valid
DO NOT use these field names: description, explanation, evidence_summary

ONLY use: answer, confidence, reason
Remember: confidence must be a NUMBER like 0.75, NOT true/false."""

        print(f"    [DEBUG] Prompt to LLM (first 500 chars):\n{formatted_base_prompt[:500]}...")

        enhanced_system_prompt = system_prompt + """

===CRITICAL FORMAT REQUIREMENT for "reason" field===
Your "reason" field MUST follow this EXACT step-by-step format:
"Step 0: [analysis]. Step 1: [analysis]. Step 2: [analysis]. Step 3: [analysis]. Step 4: [calculation]. Step 5: [fusion]. Step 6: [final decision]. Final: answer=X, confidence=Y"

DO NOT write free-form explanations. You MUST show each step explicitly."""

        response = await llm_func(
            formatted_base_prompt,
            system_prompt=enhanced_system_prompt,
            response_format={"type": "json_object"},
            max_tokens=6000
        )
        response_lower = response.lower()

        print(f"    [Graph+CoT] LLM推理完成")
        # 调试：打印LLM响应的前200字符
        print(f"    [DEBUG] LLM完整响应:\n{response}")

        llm_score = 0.0
        relation_type = "unknown"
        llm_says_yes = False

        # 首先尝试解析JSON格式（来自用户的prompt配置）
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith('```'):
                lines = cleaned_response.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].startswith('```'):
                    lines = lines[:-1] 
                cleaned_response = '\n'.join(lines)

            # 尝试解析JSON
            json_result = json.loads(cleaned_response)
            if isinstance(json_result, dict):
                # 解析成功！支持多种字段名格式

                # 优先检查answer字段（显式的yes/no答案）
                llm_says_yes = False
                json_answer_valid = False
                if 'answer' in json_result:
                    answer_val = json_result['answer']
                    if isinstance(answer_val, bool):
                        llm_says_yes = answer_val
                        json_answer_valid = True
                    elif isinstance(answer_val, str):
                        answer_norm = answer_val.strip().lower()
                        if answer_norm in ('yes', 'y', 'true', 'no', 'n', 'false'):
                            llm_says_yes = answer_norm in ('yes', 'y', 'true')
                            json_answer_valid = True
                        elif answer_norm:
                            raise InvalidLLMVerdictError(f"JSON answer字段取值非法: {answer_val!r}")

                # 提取confidence/分数 - 支持多种字段名
                llm_score = 0.0
                json_confidence_present = False
                if 'confidence' in json_result:
                    llm_score = float(json_result['confidence'])
                    json_confidence_present = True
                elif 'score' in json_result:
                    llm_score = float(json_result['score'])
                    json_confidence_present = True
                elif 'probability' in json_result:
                    llm_score = float(json_result['probability'])
                    json_confidence_present = True
                # 布尔值字段转换为分数
                else:
                    for legacy_field in ('relationship_exists', 'has_relation', 'is_valid'):
                        if legacy_field in json_result:
                            val = json_result[legacy_field]
                            if isinstance(val, bool):
                                llm_score = 1.0 if val else 0.0
                            elif isinstance(val, str):
                                llm_score = 1.0 if val.lower() in ['yes', 'true', 'y'] else 0.0
                            elif isinstance(val, (int, float)):
                                llm_score = float(val)
                            json_confidence_present = True
                            break

                # 如果没有answer字段，根据confidence判断
                if 'answer' not in json_result:
                    llm_says_yes = llm_score > 0.5

                # 提取reason/描述 - 支持多种字段名
                reason = ''
                for field in ['reason', 'description', 'explanation', 'thinking', 'analysis']:
                    if field in json_result and json_result[field]:
                        reason = str(json_result[field])
                        break

                # ===从reason的Final部分提取answer和confidence===
                # 如果reason中包含"Final: answer=X, confidence=Y"，优先使用这个值
                # 因为reason中的分析是LLM的真正思考过程，比顶层的answer字段更可靠
                final_match = re.search(r'Final:\s*answer=\s*(yes|no),\s*confidence=\s*([0-9]+(?:\.[0-9]+)?)', reason, re.IGNORECASE)
                if final_match:
                    final_answer = final_match.group(1).lower()
                    final_confidence = float(final_match.group(2))
                    print(f"    [Graph+CoT] 从reason的Final提取: answer={final_answer}, confidence={final_confidence}")
                    # 使用Final部分的值替代JSON字段值
                    llm_says_yes = (final_answer == 'yes')
                    llm_score = final_confidence
                    print(f"    [Graph+CoT] 使用reason中的Final判断（覆盖JSON字段）")
                else:
                    # 如果reason中没有Final格式，检查是否是立即拒绝的情况
                    immediate_rejection = re.search(r'immediate rejection.*?confidence\s*([0-9]+(?:\.[0-9]+)?)', reason, re.IGNORECASE)
                    if immediate_rejection:
                        reject_conf = float(immediate_rejection.group(1))
                        print(f"    [Graph+CoT] 检测到立即拒绝，使用confidence={reject_conf}")
                        llm_says_yes = False
                        llm_score = reject_conf
                        print(f"    [Graph+CoT] 使用立即拒绝判断")

                # 确保返回的分数与answer一致（矛盾时强制对齐：no→0.2, yes→0.8）
                llm_score = _reconcile_score_with_answer(llm_says_yes, llm_score)

                # 有效性门：必须有可校验的answer来源与confidence来源，且confidence∈[0,1]
                # （{} 等伪成功回答在此按失败上抛，禁止按 NO 计分）
                has_answer_source = json_answer_valid or final_match or immediate_rejection
                has_confidence_source = json_confidence_present or final_match or immediate_rejection
                if not (has_answer_source and has_confidence_source):
                    raise InvalidLLMVerdictError(
                        f"JSON回答缺少可校验的answer/confidence字段: {response[:200]}"
                    )
                if not (math.isfinite(llm_score) and 0.0 <= llm_score <= 1.0):
                    raise InvalidLLMVerdictError(f"confidence 越界[0,1]: {llm_score}")

                print(f"    [Graph+CoT] JSON格式解析成功: answer={json_result.get('answer', 'N/A')}, confidence={llm_score:.2f} ({'YES' if llm_says_yes else 'NO'})")

                return llm_score, response
        except (json.JSONDecodeError, ValueError) as e:
            # JSON解析失败，继续使用CoT格式解析
            print(f"    [Graph+CoT] JSON解析失败，使用CoT格式解析")

        # 解析LLM响应（CoT格式）
        # 首先在全文中搜索关系判断

        # 优先提取末尾的Final结论（与JSON路径同规则）：裸文本响应中推理过程
        # 可能同时含yes/no关键词、且Step 3的"average confidence"会早于真正结论
        # 出现，关键词启发式与首个confidence正则都可能误判（如把0.10当置信度）
        final_match = re.search(r'Final:\s*answer=\s*(yes|no),\s*confidence=\s*([0-9]+(?:\.[0-9]+)?)', response, re.IGNORECASE)
        if final_match:
            llm_says_yes = final_match.group(1).lower() == 'yes'
            llm_score = float(final_match.group(2))
            print(f"    [Graph+CoT] CoT回退解析提取Final: answer={final_match.group(1)}, confidence={llm_score:.2f}")
            # 与JSON路径一致的分数-answer一致性修正
            llm_score = _reconcile_score_with_answer(llm_says_yes, llm_score)
            if not (math.isfinite(llm_score) and 0.0 <= llm_score <= 1.0):
                raise InvalidLLMVerdictError(f"confidence 越界[0,1]: {llm_score}")
            relation_type = _infer_relation_type(response_lower)
            print(f"    [Graph+CoT] LLM分数: {float(llm_score):.2f}, 关系类型: {relation_type}, 判断: {'YES' if llm_says_yes else 'NO'}")
            return llm_score, response

        # 多种格式的关系判断提取（仅显式格式算有效回答；开头词/关键词等模糊启发式
        # 不作为有效性依据——伪成功文本（如"Unable to process request."）必须按失败处理）
        # 格式1: STEP4_JUDGMENT: [RELATIONSHIP: YES/NO]
        explicit_answer = False
        step4_match = re.search(r'STEP4_JUDGMENT:.*?RELATIONSHIP:\s*(YES|NO)', response, re.IGNORECASE)
        if step4_match:
            llm_says_yes = step4_match.group(1).upper() == 'YES'
            explicit_answer = True
            print(f"    [Graph+CoT] LLM判断: RELATIONSHIP = {step4_match.group(1).upper()} (STEP4格式)")
        else:
            # 格式2: meaningful medical relationship: YES/NO
            meaningful_match = re.search(r'meaningful medical relationship:\s*\*?\*?\s*(YES|NO)', response, re.IGNORECASE)
            if meaningful_match:
                llm_says_yes = meaningful_match.group(1).upper() == 'YES'
                explicit_answer = True
                print(f"    [Graph+CoT] LLM判断: RELATIONSHIP = {meaningful_match.group(1).upper()} (meaningful relationship格式)")
            else:
                # 格式3: Final Answer: **Yes/No**
                final_answer_match = re.search(r'Final Answer:\s*\*?\*?\s*(YES|NO)', response, re.IGNORECASE)
                if final_answer_match:
                    llm_says_yes = final_answer_match.group(1).upper() == 'YES'
                    explicit_answer = True
                    print(f"    [Graph+CoT] LLM判断: RELATIONSHIP = {final_answer_match.group(1).upper()} (Final Answer格式)")
                else:
                    # 格式4: relationship is/exist: YES/NO
                    relationship_exist_match = re.search(r'relationship\s+(is|exists?)\s*:\s*(YES|NO)', response, re.IGNORECASE)
                    if relationship_exist_match:
                        llm_says_yes = relationship_exist_match.group(2).upper() == 'YES'
                        explicit_answer = True
                        print(f"    [Graph+CoT] LLM判断: RELATIONSHIP = {relationship_exist_match.group(2).upper()} (relationship exists格式)")
                    else:
                        # 格式5: 检查响应开头的yes/no（仅打印参考，不作为有效回答依据）
                        first_words = response_lower.strip()[:30]
                        if first_words.startswith('yes'):
                            llm_says_yes = True
                            print(f"    [Graph+CoT] LLM判断: RELATIONSHIP = YES (开头匹配)")
                        elif first_words.startswith('no'):
                            llm_says_yes = False
                            print(f"    [Graph+CoT] LLM判断: RELATIONSHIP = NO (开头匹配)")
                        # 如果开头不明确，检查关键词
                        elif 'yes' in response_lower and 'no' not in response_lower:
                            llm_says_yes = True
                            print(f"    [Graph+CoT] LLM判断: RELATIONSHIP = YES (关键词匹配)")
                        elif 'no' in response_lower and 'yes' not in response_lower:
                            llm_says_yes = False
                            print(f"    [Graph+CoT] LLM判断: RELATIONSHIP = NO (关键词匹配)")

        # 提取置信度（confidence 直接表示"关系成立的概率"，不需要反转）
        confidence_match = re.search(r'confidence[:\s]+([0-9]+(?:\.[0-9]+)?)', response, re.IGNORECASE)
        if confidence_match:
            llm_score = float(confidence_match.group(1))
            print(f"    [Graph+CoT] 置信度: {llm_score:.2f} ({'YES' if llm_score > 0.5 else 'NO'})")

        # 有效性门：显式yes/no与显式confidence二者必备，越界按失败上抛（缺失即按失败上抛）
        if not explicit_answer:
            raise InvalidLLMVerdictError(f"回答中未找到明确的yes/no判断: {response[:200]}")
        if not confidence_match:
            raise InvalidLLMVerdictError(f"回答缺少confidence数值: {response[:200]}")
        if not (0.0 <= llm_score <= 1.0):
            raise InvalidLLMVerdictError(f"confidence 越界[0,1]: {llm_score}")

        relation_type = _infer_relation_type(response_lower)

        print(f"    [Graph+CoT] LLM分数: {float(llm_score):.2f}, 关系类型: {relation_type}, 判断: {'YES' if llm_says_yes else 'NO'}")

        return llm_score, response

    except Exception:
        # 验证调用失败必须上抛：吞掉会把API故障记成0分"无关系"假记录，
        # 污染断点续传（失败本应不落盘）与最终统计
        raise



async def _filter_documents_by_relevance(
    documents: list,
    query_entity_a: str,
    query_entity_b: str,
    embedding_func_config: dict = None,
    min_relevance: float = 0.5,
    min_docs: int = 2
) -> tuple:
    """
    对完整摘要进行相关性预过滤（上游噪声切除）

    策略：
    1. 计算每篇摘要与查询的相关性分数
    2. 应用阈值过滤（默认0.5）
    3. **最多保留10篇摘要**（PubMed检索的原始上限）

    Args:
        documents: 完整摘要列表，每个元素为 {'pmid': xxx, 'abstract': xxx}
        query_entity_a: 查询实体A
        query_entity_b: 查询实体B
        embedding_func_config: embedding配置
        min_relevance: 最小相关性阈值（默认0.5）
        min_docs: 最少保留文档数（低于此值不使用优先级1）

    Returns:
        (filtered_documents, stats)
        - filtered_documents: 过滤后的文档列表（最多10篇），如果不足min_docs篇则返回空列表
        - stats: 统计信息字典
    """
    if not documents:
        return [], {'total': 0, 'filtered': 0, 'kept': 0, 'threshold_used': 0.0}

    print(f"    [摘要预过滤] 开始过滤 {len(documents)} 篇摘要...")

    # === 步骤1：生成图谱专用查询（宽泛的，覆盖多种关系类型）===
    # 这个查询比LLM查询更宽泛，用于评估摘要是否包含任何相关信息
    graph_query = (
        f"What are the relationships between {query_entity_a} and {query_entity_b}, "
        f"including treatments, causes, pharmacological effects, receptor-ligand interactions, "
        f"enzyme-substrate reactions, drug-disease associations, and genetic correlations?"
    )

    # === 步骤2：计算摘要相关性分数 ===
    doc_scores = []

    try:
        # 获取embedding函数
        if embedding_func_config and 'func' in embedding_func_config:
            embed_func = partial(
                embedding_func_config['func'],
                model=embedding_func_config['model'],
                api_key=embedding_func_config['api_key'],
                base_url=embedding_func_config['base_url']
            )

            # 向量化查询
            query_embedding = await embed_func([graph_query])
            if query_embedding is None or len(query_embedding) == 0:
                raise Exception("Query embedding failed")
            query_vec = query_embedding[0] if hasattr(query_embedding[0], '__iter__') else query_embedding

            # 向量化每篇摘要
            abstract_texts = [doc.get('abstract', '') for doc in documents]
            doc_embeddings = await embed_func(abstract_texts)

            if doc_embeddings is None or len(doc_embeddings) == 0:
                raise Exception("Document embeddings failed")

            # 计算向量相似度
            for i, (doc, doc_emb) in enumerate(zip(documents, doc_embeddings)):
                doc_vec = doc_emb if hasattr(doc_emb, '__iter__') else [doc_emb]
                similarity = float(np.dot(query_vec, doc_vec) /
                                 (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-8))
                doc_scores.append({
                    'doc': doc,
                    'similarity': similarity,
                    'rerank_score': None,  # 暂时不用rerank（需要额外API调用）
                    'combined_score': similarity  # 先用向量相似度
                })
        else:
            # 没有embedding配置，使用简单的关键词匹配作为回退
            query_lower = graph_query.lower()
            entity_lower = f"{query_entity_a.lower()} {query_entity_b.lower()}".lower()

            for doc in documents:
                abstract_lower = doc.get('abstract', '').lower()
                # 计算关键词出现频率
                entity_a_count = abstract_lower.count(query_entity_a.lower())
                entity_b_count = abstract_lower.count(query_entity_b.lower())
                total_mentions = entity_a_count + entity_b_count

                # 简单的相关性评分：0-1范围
                score = min(1.0, total_mentions / 10.0) if total_mentions > 0 else 0.0

                doc_scores.append({
                    'doc': doc,
                    'similarity': score,
                    'rerank_score': None,
                    'combined_score': score
                })

    except Exception as e:
        print(f"    [摘要预过滤] 计算相关性失败: {e}，使用全部文档")
        return documents, {
            'total': len(documents),
            'filtered': 0,
            'kept': len(documents),
            'threshold_used': 0.0,
            'error': str(e)
        }

    # === 步骤3：应用阈值过滤 ===
    # 排序
    doc_scores.sort(key=lambda x: x['combined_score'], reverse=True)

    # 统计
    high_relevance = [s for s in doc_scores if s['combined_score'] >= min_relevance]

    # === 步骤4：数量限制 ===
    # 限制最多保留10篇摘要（PubMed检索的上限）
    MAX_DOCS_TO_KEEP = 10

    threshold_used = min_relevance
    filtered_docs = [s['doc'] for s in high_relevance]

    # 检查是否有足够的高分文档
    if len(filtered_docs) < min_docs:
        # 高分文档不足，返回空列表让调用方回退到其他优先级
        print(f"    [摘要预过滤] 高分文档不足({len(filtered_docs)} < {min_docs})，回退到其他优先级")
        return [], {
            'total': len(documents),
            'filtered': len(documents),
            'kept': 0,
            'threshold_used': threshold_used,
            'avg_score': sum(s['combined_score'] for s in doc_scores) / len(doc_scores),
            'scores': [s['combined_score'] for s in doc_scores[:5]],
            'insufficient': True
        }

    # 最多保留10篇
    if len(filtered_docs) > MAX_DOCS_TO_KEEP:
        original_count = len(filtered_docs)
        filtered_docs = filtered_docs[:MAX_DOCS_TO_KEEP]
        print(f"    [摘要预过滤] 限制数量：从 {original_count} 篇减少到 {MAX_DOCS_TO_KEEP} 篇")

    # 打印统计信息
    stats = {
        'total': len(documents),
        'filtered': len(documents) - len(filtered_docs),
        'kept': len(filtered_docs),
        'threshold_used': threshold_used,
        'avg_score': sum(s['combined_score'] for s in doc_scores) / len(doc_scores),
        'scores': [s['combined_score'] for s in doc_scores[:5]]  # 前5名分数
    }

    print(f"    [摘要预过滤] 原始: {stats['total']} 篇")
    print(f"    [摘要预过滤] 过滤掉: {stats['filtered']} 篇（噪声）")
    print(f"    [摘要预过滤] 保留: {stats['kept']} 篇（高质量）")
    print(f"    [摘要预过滤] 阈值: {threshold_used:.2f}, 平均分: {stats['avg_score']:.3f}")
    print(f"    [摘要预过滤] 前3名分数: {[f'{s:.3f}' for s in stats['scores'][:3]]}")

    return filtered_docs, stats


async def build_minimal_viable_temp_graph(
    pubmed_data: dict,
    llm_func,
    embedding_func_config: dict = None,
    query_entity_a: str = "",
    query_entity_b: str = "",
    entity_a_standard: str = None,
    entity_a_cui: str = None,
    entity_b_standard: str = None,
    entity_b_cui: str = None
):
    """
    基于PubMed摘要构建临时图谱

    Returns:
        临时LightRAG实例
    """
    temp_dir = tempfile.mkdtemp(prefix="temp_graph_", dir=str(TEMP_DIR_BASE))
    print(f"    [临时图谱] 创建目录: {temp_dir}")

    target_entities = []
    if query_entity_a:
        target_entities.append(query_entity_a)
    if query_entity_b:
        target_entities.append(query_entity_b)

    if embedding_func_config:
        temp_rag = LightRAG(
            working_dir=temp_dir,
            workspace=f"ws_{uuid.uuid4().hex[:8]}",
            llm_model_func=llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=partial(
                    embedding_func_config['func'],
                    model=embedding_func_config['model'],
                    api_key=embedding_func_config['api_key'],
                    base_url=embedding_func_config['base_url'],
                ),
            ),
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            max_parallel_insert=2,
            addon_params={
                "language": "English",
                "entity_types": ["Disease", "ChemicalOrDrug", "GeneOrProtein", "Symptom", "BiologicalProcess"],
                "target_entities": target_entities
            }
        )
    else:
        temp_rag = LightRAG(
            working_dir=temp_dir,
            workspace=f"ws_{uuid.uuid4().hex[:8]}",
            llm_model_func=llm_func,
            chunk_token_size=1200,
            max_parallel_insert=2,
            addon_params={
                "language": "English",
                "entity_types": ["Disease", "ChemicalOrDrug", "GeneOrProtein", "Symptom", "BiologicalProcess"],
                "target_entities": target_entities
            }
        )

    await temp_rag.initialize_storages()
    print(f"    [临时图谱] 存储初始化完成")
    # === 图谱构建优先级 ===
    #
    # 优先级1 (两臂同架构): 按检索层精排结果(retrieval_results)的PMID选取对应完整摘要
    #   - 数据源: 精排选中PMID的完整摘要（去重，按精排序取前10篇，允许单篇）
    #   - 理由: 让检索/精排的差异真正传导进图谱构建（baseline与hybrid共用本逻辑，
    #     唯一差异=召回排序），并记录实际建图PMID与文本哈希供核查
    #
    # 优先级2 (回退A): 精排结果无可用摘要时，退回摘要预过滤路径(向量相似度>=0.5且>=2篇)
    # 优先级3 (回退B): 预过滤也不足时，使用Reranked chunks(>=3个)

    documents = pubmed_data.get('documents', [])
    retrieval_results = pubmed_data.get('retrieval_results', []) if pubmed_data else []
    MAX_DOCS_TO_KEEP = 10
    data_quality = "unknown"
    graph_build_sources = []

    # === 优先级1: 精排选中的完整摘要 ===
    if retrieval_results:
        pmid_order = []
        for chunk_item in retrieval_results:
            pmid = str((chunk_item.get('metadata') or {}).get('pmid', '') or '').strip()
            if pmid and pmid != 'N/A' and pmid not in pmid_order:
                pmid_order.append(pmid)
        pmid_order = pmid_order[:MAX_DOCS_TO_KEEP]

        docs_by_pmid = {}
        for doc in documents:
            doc_pmid = str(doc.get('pmid', '') or '').strip()
            if doc_pmid and doc_pmid not in docs_by_pmid:
                docs_by_pmid[doc_pmid] = doc

        selected_docs = [docs_by_pmid[p] for p in pmid_order if p in docs_by_pmid]

        if selected_docs:
            print(f"    [临时图谱] 优先级1: 按精排结果建图 ({len(selected_docs)} 篇完整摘要, 精排池 {len(pmid_order)} PMIDs)")
            for doc in selected_docs:
                source_text = doc.get('abstract', '') if isinstance(doc, dict) else str(doc)
                if not source_text:
                    continue
                track_id = await temp_rag.ainsert(source_text)
                pmid = str(doc.get('pmid', 'N/A'))
                text_sha1 = hashlib.sha1(source_text.encode('utf-8')).hexdigest()[:16]
                graph_build_sources.append(f"{pmid}:{text_sha1}")
                print(f"    [临时图谱] 插入摘要 track_id={track_id}, PMID={pmid}, sha1={text_sha1}")

            if graph_build_sources:
                data_quality = f"priority1_rerank_selected_abstracts({len(graph_build_sources)}/{len(pmid_order)})"
        else:
            print(f"    [临时图谱] 精排PMID在完整摘要集中无匹配，回退到优先级2")

    # === 优先级2 (回退A): 完整摘要预过滤 ===
    if data_quality == "unknown":
        if documents:
            print(f"    [临时图谱] 尝试使用完整摘要构建 ({len(documents)} 篇)")

            # 进行相关性预过滤
            filtered_docs, filter_stats = await _filter_documents_by_relevance(
                documents=documents,
                query_entity_a=query_entity_a,
                query_entity_b=query_entity_b,
                embedding_func_config=embedding_func_config,
                min_relevance=0.5,
                min_docs=2
            )

            # 检查过滤后是否满足条件
            if len(filtered_docs) >= 2:
                print(f"    [临时图谱] 优先级2: 使用过滤后摘要 ({filter_stats['kept']}/{filter_stats['total']}, 阈值={filter_stats['threshold_used']:.2f})")

                for doc in filtered_docs:
                    source_text = doc.get('abstract', '') if isinstance(doc, dict) else str(doc)
                    track_id = await temp_rag.ainsert(source_text)
                    pmid = doc.get('pmid', 'N/A')
                    text_sha1 = hashlib.sha1(source_text.encode('utf-8')).hexdigest()[:16]
                    graph_build_sources.append(f"{pmid}:{text_sha1}")
                    print(f"    [临时图谱] 插入摘要 track_id={track_id}, PMID={pmid}, sha1={text_sha1}")

                data_quality = f"priority2_filtered_abstracts({filter_stats['kept']}/{filter_stats['total']})"
            else:
                print(f"    [临时图谱] 过滤后摘要不足({len(filtered_docs)}<2)，回退到优先级3")
                # 继续到优先级3
        else:
            print(f"    [临时图谱] 无完整摘要，直接进入优先级3")

    # === 优先级3 (回退B): Reranked chunks（无回退） ===
    if data_quality == "unknown":
        reranked_chunks = retrieval_results

        if reranked_chunks and len(reranked_chunks) >= 3:
            # 使用Reranked chunks（经过混合评分精排）
            print(f"    [临时图谱] 优先级3: 使用Reranked chunks ({len(reranked_chunks)} 个)")

            for idx, chunk_item in enumerate(reranked_chunks):
                chunk_text = chunk_item.get('chunk_text', '')
                if chunk_text:
                    track_id = await temp_rag.ainsert(chunk_text)
                    pmid = chunk_item.get('metadata', {}).get('pmid', 'N/A')
                    combined_score = chunk_item.get('combined_score', 0.0)
                    print(f"    [临时图谱] 插入Reranked chunk [{idx+1}] track_id={track_id}, PMID={pmid}, score={combined_score:.3f}")

            data_quality = f"priority3_reranked_chunks({len(reranked_chunks)})"
        else:
            # Rerank失败或chunks不足，不构建图谱，直接判断无关系
            print(f"    [临时图谱] Reranked chunks不足 ({len(reranked_chunks) if reranked_chunks else 0} < 3)，不构建图谱")
            data_quality = "no_valid_chunks"

    # 记录实际建图来源（PMID:文本sha1前16位），供调用方写入结果与核查
    temp_rag._graph_build_sources = graph_build_sources
    temp_rag._graph_data_quality = data_quality

    return temp_rag


# ============== 核心查询函数 ==============

async def prepare_query_context(
    entity_a: str,
    entity_b: str,
    abstract: str = "",
    enable_rerank: bool = True
) -> dict:
    """图谱构建前的检索上下文：PubMed动态检索 + MeSH规范化 + QueryDocument

    从enhanced_query_entity_relation中分离，供消融实验等需要"建图一次、多配置复用"的场景使用。
    """
    entity_a_standard = entity_a
    entity_b_standard = entity_b

    print(f"  [动态RAG] 获取PubMed文献...")
    dynamic_rag = EnhancedDynamicRAG(max_pmids=10, use_cache=True, enable_rerank=enable_rerank)
    pubmed_data = await dynamic_rag.build_rag_for_query(
        entity_a_standard,
        entity_b_standard,
        "Disease", "Disease",
        ""
    )

    # 提取MeSH术语用于规范化实体名
    mesh_term_a, mesh_term_b = pubmed_data.get('mesh_terms', (None, None))
    if mesh_term_a and mesh_term_a != entity_a:
        print(f"  [图谱查询] 实体A: '{entity_a_standard}' -> '{mesh_term_a}' (MeSH提取)")
        entity_a_standard = mesh_term_a
    if mesh_term_b and mesh_term_b != entity_b:
        print(f"  [图谱查询] 实体B: '{entity_b_standard}' -> '{mesh_term_b}' (MeSH提取)")
        entity_b_standard = mesh_term_b

    # QueryDocument：优先使用传入的abstract，否则拼接检索摘要前3篇
    if abstract and abstract.strip():
        query_document = abstract.strip()
        print(f"  [QueryDocument] 使用传入的abstract作为Query Document (长度: {len(abstract)} 字符)")
    else:
        pubmed_abstracts = []
        for doc in pubmed_data.get('documents', [])[:3]:
            abs_text = doc.get('abstract', '')
            if abs_text:
                pubmed_abstracts.append(f"[PMID:{doc.get('pmid', 'unknown')}] {abs_text}")
        query_document = "\n\n".join(pubmed_abstracts[:3])
        if pubmed_abstracts:
            print(f"  [QueryDocument] 使用 {len(pubmed_abstracts[:3])} 篇检索摘要作为Query Document")

    return {
        'pubmed_data': pubmed_data,
        'query_document': query_document,
        'entity_a_standard': entity_a_standard,
        'entity_b_standard': entity_b_standard
    }


async def enhanced_query_entity_relation(
    rag: LightRAG,
    entity_a: str,
    entity_b: str,
    llm_func,
    working_dir: str = None,
    abstract: str = "",
    track_id: str = None,
    use_rag: bool = True,
    enable_rerank: bool = True,
    entity_a_type: str = "",
    entity_b_type: str = "",
    max_hops: int = 2,
    lambda_table: dict = None,
    prebuilt_graph=None,
    enable_nl_fallback: bool = True
) -> dict:
    """
    基于临时图谱的实体关系查询函数（简化版）

    Returns:
        {
            'entity_a': str, 'entity_b': str,
            'has_relation': bool,
            'relation_type': str, 'confidence': float,
            'reason': str,
            'multi_hop_path': list,
            'pubmed_context': str,
            'graph_score': float,
            'llm_score': float
        }
    """
    print(f"\n[临时图谱查询] 查询 '{entity_a}' 和 '{entity_b}' 之间的关系...")

    entity_a_cui = None
    entity_b_cui = None

    temp_graph = None

    try:
        # === 阶段1: PubMed检索 + MeSH规范化 + QueryDocument ===
        context = await prepare_query_context(
            entity_a, entity_b,
            abstract=abstract,
            enable_rerank=enable_rerank
        )
        pubmed_data = context['pubmed_data']
        query_document = context['query_document']
        entity_a_standard = context['entity_a_standard']
        entity_b_standard = context['entity_b_standard']

        all_chunks = pubmed_data.get('chunks', []) if pubmed_data else []
        print(f"  [动态RAG] 获取了 {len(all_chunks)} 个chunks (用于图谱构建)")

        # === 阶段3: 构建临时图谱 ===
        embedding_config = {
            'func': openai_embed.func,
            'model': QWEN_EMBEDDING_MODEL,
            'api_key': QWEN_API_KEY,
            'base_url': QWEN_BASE_URL
        }

        if prebuilt_graph is not None:
            print(f"  [临时图谱] 使用预构建图谱（跳过构建）")
            temp_graph = prebuilt_graph
        else:
            print(f"  [临时图谱] 构建规范化图谱...")
            temp_graph = await build_minimal_viable_temp_graph(
                pubmed_data=pubmed_data,
                llm_func=llm_func,
                embedding_func_config=embedding_config,
                query_entity_a=entity_a_standard if entity_a_standard else entity_a,
                query_entity_b=entity_b_standard if entity_b_standard else entity_b,
                entity_a_standard=entity_a_standard,
                entity_a_cui=entity_a_cui,
                entity_b_standard=entity_b_standard,
                entity_b_cui=entity_b_cui
            )

        # === 阶段4: 多跳查询临时图谱 ===
        print(f"  [临时图谱] 执行多跳查询...")

        # 检查是否使用原始实体名称（MeSH匹配失败时）
        use_original_a = pubmed_data.get('use_original_entities', [False, False])[0] if pubmed_data else False
        use_original_b = pubmed_data.get('use_original_entities', [False, False])[1] if pubmed_data else False

        # 如果MeSH匹配失败，使用原始实体名称
        if use_original_a:
            entity_a_for_query = entity_a
            print(f"  [临时图谱] MeSH匹配失败（实体A），使用原始名称: '{entity_a}'")
        else:
            entity_a_for_query = entity_a_standard if entity_a_standard else entity_a

        if use_original_b:
            entity_b_for_query = entity_b
            print(f"  [临时图谱] MeSH匹配失败（实体B），使用原始名称: '{entity_b}'")
        else:
            entity_b_for_query = entity_b_standard if entity_b_standard else entity_b

        print(f"  [临时图谱] 查询实体: A='{entity_a_for_query}', B='{entity_b_for_query}'")

        temp_relation = None
        multi_hop_path = []

        try:
            if entity_a_for_query and entity_b_for_query:
                print(f"  [临时图谱] 尝试1: 使用规范化名称查询...")
                temp_relation = await temp_graph.get_relation_info(entity_a_for_query, entity_b_for_query)

                if temp_relation and temp_relation.get('graph_data'):
                    print(f"  [临时图谱] [OK] 找到关系（规范化名称）")
                else:
                    print(f"  [临时图谱] 尝试2: 使用原始名称查询...")
                    temp_relation = await temp_graph.get_relation_info(entity_a, entity_b)

                    if temp_relation and temp_relation.get('graph_data'):
                        print(f"  [临时图谱] [OK] 找到关系（原始名称）")
                    else:
                        print(f"  [临时图谱] 尝试3: 混合查询（A标准+B原始）...")
                        temp_relation = await temp_graph.get_relation_info(entity_a_for_query, entity_b)

                        if temp_relation and temp_relation.get('graph_data'):
                            print(f"  [临时图谱] [OK] 找到关系（混合）")
                        else:
                            print(f"  [临时图谱] 尝试4: 混合查询（A原始+B标准）...")
                            temp_relation = await temp_graph.get_relation_info(entity_a, entity_b_for_query)

                            if temp_relation and temp_relation.get('graph_data'):
                                print(f"  [临时图谱] [OK] 找到关系（混合）")

            has_valid_relation = False
            if temp_relation and temp_relation.get('graph_data'):
                graph_data = temp_relation['graph_data']

                if isinstance(graph_data, dict):
                    weight = graph_data.get('weight', 0)
                    description = graph_data.get('description', '')
                    keywords = graph_data.get('keywords', '')

                    # DEBUG: 打印边的详细信息
                    print(f"  [DEBUG] 边的详细信息:")
                    print(f"    - weight: {weight}")
                    print(f"    - keywords: '{keywords}'")
                    print(f"    - description: '{description[:200]}...' (完整长度: {len(description)})")

                    # 基本条件检查（由LightRAG prompt负责过滤无效关系）
                    if weight > 0 or (description and len(description) > 10):
                        has_valid_relation = True
                        print(f"  [临时图谱] [OK] 找到有效关系 (weight={weight}, desc_len={len(description)})")

                if has_valid_relation:
                    actual_a = await _ensure_graph_node_name(entity_a_for_query, temp_graph)
                    actual_b = await _ensure_graph_node_name(entity_b_for_query, temp_graph)
                    multi_hop_path = [actual_a, actual_b]
                    print(f"  [临时图谱] 直接关系查询成功")
                    print(f"  [调试Graph] 设置路径: [{actual_a}, {actual_b}] (查询名: [{entity_a_for_query}, {entity_b_for_query}])")

            if not has_valid_relation:
                print(f"  [临时图谱] 精确匹配失败，尝试模糊匹配...")

                fuzzy_a = await _find_similar_entity_in_graph(entity_a_for_query, temp_graph)
                fuzzy_b = await _find_similar_entity_in_graph(entity_b_for_query, temp_graph)

                if fuzzy_a and fuzzy_b:
                    print(f"  [临时图谱] 模糊匹配成功: '{entity_a_for_query}'->'{fuzzy_a}', '{entity_b_for_query}'->'{fuzzy_b}'")

                    temp_relation = await temp_graph.get_relation_info(fuzzy_a, fuzzy_b)

                    if temp_relation and temp_relation.get('graph_data'):
                        graph_data = temp_relation['graph_data']
                        weight = graph_data.get('weight', 0)
                        description = graph_data.get('description', '')
                        keywords = graph_data.get('keywords', '')

                        # DEBUG: 打印边的详细信息（模糊匹配）
                        print(f"  [DEBUG] 模糊匹配边的详细信息:")
                        print(f"    - 匹配实体: '{fuzzy_a}' <-> '{fuzzy_b}'")
                        print(f"    - weight: {weight}")
                        print(f"    - keywords: '{keywords}'")
                        print(f"    - description: '{description[:200]}...' (完整长度: {len(description)})")
                        print(f"    - 验证条件: weight>0={weight>0}, desc_len>10={len(description)>10}")

                        if weight > 0 or (description and len(description) > 10):
                            has_valid_relation = True
                            print(f"  [临时图谱] [OK] 模糊匹配找到有效关系")
                            multi_hop_path = [fuzzy_a, fuzzy_b]

                if not has_valid_relation:
                    print(f"  [临时图谱] 未找到直接关系，尝试图谱算法多跳查询...")

                    query_a = fuzzy_a if fuzzy_a else entity_a_for_query
                    query_b = fuzzy_b if fuzzy_b else entity_b_for_query

                    multi_hop_path = await _find_multi_hop_path_in_graph(query_a, query_b, temp_graph, max_hops=max_hops)

                    if multi_hop_path and len(multi_hop_path) > 2:
                        has_valid_relation = True
                        print(f"  [临时图谱] [OK] 多跳路径查询成功，路径长度: {len(multi_hop_path)}")
                    elif enable_nl_fallback:
                        print(f"  [临时图谱] 图谱算法未找到多跳路径，尝试自然语言查询...")
                        query_text = f"What is the relationship path between {entity_a_for_query} and {entity_b_for_query}? Include any intermediate entities."

                        try:
                            query_result = await temp_graph.aquery(
                                query_text,
                                param=QueryParam(mode="local", only_need_context=False)
                            )

                            if hasattr(query_result, '__aiter__'):
                                result_parts = []
                                async for part in query_result:
                                    result_parts.append(part)
                                query_result = ''.join(result_parts)

                            if query_result:
                                llm_multi_hop_path = _extract_multi_hop_path(query_result)
                                if llm_multi_hop_path:
                                    corrected_path = []
                                    for node in llm_multi_hop_path:
                                        actual_node = await _ensure_graph_node_name(node, temp_graph)
                                        corrected_path.append(actual_node)
                                    # 与NetworkX分支保持一致的跳数约束
                                    if (len(corrected_path) - 1) <= max_hops:
                                        multi_hop_path = corrected_path
                                        print(f"  [临时图谱] [OK] 自然语言多跳查询成功，路径长度: {len(multi_hop_path)}")
                                    else:
                                        print(f"  [临时图谱] 自然语言路径长度 {len(corrected_path)-1} 跳超过 max_hops={max_hops}，丢弃")

                        except Exception as e:
                            print(f"  [警告] 自然语言查询失败: {e}")

        except Exception as e:
            print(f"  [警告] 临时图谱查询异常: {e}")
            import traceback
            traceback.print_exc()

        # 调试：显示multi_hop_path的内容
        print(f"  [调试Graph] multi_hop_path: {multi_hop_path}")

        graph_score = await _calculate_graph_score(multi_hop_path, temp_graph, lambda_table=lambda_table)

        if graph_score > 0:
            print(f"  [临时图谱] Graph分数: {float(graph_score):.2f}")
        else:
            print(f"  [临时图谱] 未找到有效关系路径，Graph分数为0")

        # === 阶段7: LLM验证（RAG + CoT） ===
        # 构建知识图谱信息（用于RAG增强）
        knowledge_graph = ""
        if temp_graph and multi_hop_path and len(multi_hop_path) >= 2:
            try:
                # 获取边数据（避免重复获取）
                _, _, _, _, edge_data_list = await _validate_path_semantics(multi_hop_path, temp_graph)
                knowledge_graph = await format_path_for_llm(multi_hop_path, edge_data_list)
                print(f"    [知识图谱] 生成RAG增强信息 (路径长度: {len(multi_hop_path)}个节点)")
            except Exception as e:
                print(f"    [知识图谱] 生成失败: {e}")
                knowledge_graph = f"Knowledge Graph Path: {' -> '.join(multi_hop_path)}"
        elif multi_hop_path and len(multi_hop_path) >= 2:
            knowledge_graph = f"Knowledge Graph Path: {' -> '.join(multi_hop_path)}"

        # 调用LLM验证（使用Query Document + 知识图谱RAG + Graph-LLM Fusion）
        llm_score, llm_reason = await _llm_verify_relation_with_graph_cot(
            llm_func,
            entity_a,
            entity_b,
            multi_hop_path,
            query_document,
            knowledge_graph,  # 知识图谱RAG增强
            graph_score,     # Graph-LLM Fusion增强
            entity_a_standard,
            entity_b_standard,
            temp_graph,
            entity_a_type,
            entity_b_type
        )

        # 使用LLM分数作为最终判断（LLM已经综合了图谱路径和摘要信息）
        final_score = llm_score

        has_relation = final_score > 0.5

        return {
            'entity_a': entity_a,
            'entity_b': entity_b,
            'entity_a_standard': entity_a_standard,
            'entity_b_standard': entity_b_standard,
            'entity_a_cui': entity_a_cui,
            'entity_b_cui': entity_b_cui,
            'has_relation': has_relation,
            'relation_type': 'multi_hop' if len(multi_hop_path) > 2 else 'direct',
            'confidence': final_score,
            'reason': llm_reason,
            'multi_hop_path': multi_hop_path,
            'rag_context': '',
            'pubmed_context': query_document,  # 使用query_document（来自传入的abstract或实时检索）
            'graph_score': graph_score,
            'llm_score': llm_score,
            'graph_build_sources': getattr(temp_graph, '_graph_build_sources', []) if temp_graph is not None else [],
            'graph_data_quality': getattr(temp_graph, '_graph_data_quality', 'unknown') if temp_graph is not None else 'no_graph',
            'pmids': pubmed_data.get('pmids', [])[:10] if pubmed_data else [],
            'retrieved_chunks_count': 0
        }

    except Exception:
        print(f"  [错误] 查询失败（异常上抛，由调用方决定是否重试）")
        raise
    finally:
        # 预构建图谱归调用方管理；自建临时图谱用完即删，防止 temp_workspace 无限膨胀
        if temp_graph is not None and temp_graph is not prebuilt_graph:
            try:
                await temp_graph.finalize_storages()
            except Exception:
                pass
            shutil.rmtree(temp_graph.working_dir, ignore_errors=True)
            print(f"    [临时图谱] 已清理临时目录: {temp_graph.working_dir}")
