r"""
SyRACT-style Dynamic RAG with Persistent Cache
集成持久化缓存的动态RAG系统

改进：
1. 支持实体MeSH映射
2. 解决BRCA1、HNF-6等非标准MeSH术语的映射问题
3. 持久化缓存避免重复处理相同文献

工作流程：
1. 实体输入 → MeSH映射 → PMID检索
2. PMID → 检查缓存 → 未命中则获取摘要、切块、向量化并缓存
3. 查询 → 向量检索 → Rerank精排 → RAG结果
"""

import os
import re
import json
import time
import asyncio
import numpy as np
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment
load_dotenv(override=False)

# API Configuration
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")

# NCBI Configuration
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_TOOL = "SyRACT_RAG"
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")

# Proxy Configuration (用于访问 NCBI 等需要代理的服务)
HTTP_PROXY = os.getenv("HTTP_PROXY", "")
HTTPS_PROXY = os.getenv("HTTPS_PROXY", "")
PROXIES = {}
if HTTP_PROXY or HTTPS_PROXY:
    if HTTP_PROXY:
        PROXIES['http'] = HTTP_PROXY
    if HTTPS_PROXY:
        PROXIES['https'] = HTTPS_PROXY
    print(f"[Proxy] Using proxy: {PROXIES}")

# Cache Configuration
CACHE_DIR = Path(os.getenv("GRACE_CACHE_DIR", str(Path(__file__).resolve().parent / "dynamic_rag_cache")))
CACHE_FILE = CACHE_DIR / "pmid_cache.json"
EMBEDDING_CACHE_DIR = CACHE_DIR / "embeddings"
ENTITY_CACHE_FILE = CACHE_DIR / "entity_normalization_cache.json"

# Initialize cache directory
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)



# =============================================================================
# PMID Cache Module
# =============================================================================

class PMIDCache:
    """
    PMID文献缓存管理器

    避免重复处理相同的文献，缓存：
    - 摘要文本
    - 切块结果
    - 向量嵌入
    """

    def __init__(self):
        self.cache = {}
        self._load_cache()

    def _load_cache(self):
        """从磁盘加载缓存"""
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"[Cache] Loaded {len(self.cache)} cached PMIDs")
            except Exception as e:
                print(f"[Cache] Failed to load cache: {e}")
                self.cache = {}
        else:
            print(f"[Cache] No existing cache, starting fresh")

    def _save_cache(self):
        """保存缓存索引到磁盘"""
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                # 只保存索引，不保存embedding
                cache_to_save = {}
                for pmid, data in self.cache.items():
                    cache_to_save[pmid] = {
                        'pmid': data['pmid'],
                        'abstract': data['abstract'],
                        'chunks': data['chunks'],
                        'has_embedding': data['has_embedding'],
                        'chunk_count': data['chunk_count'],
                        'cached_at': data['cached_at'],
                        'access_count': data.get('access_count', 0)
                    }
                json.dump(cache_to_save, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Cache] Failed to save cache: {e}")

    def get(self, pmid: str) -> Optional[Dict]:
        """获取缓存的文献数据"""
        if pmid not in self.cache:
            return None

        data = self.cache[pmid]
        data['access_count'] = data.get('access_count', 0) + 1
        data['last_accessed'] = datetime.now().isoformat()

        # 加载embedding
        if data.get('has_embedding'):
            emb_file = EMBEDDING_CACHE_DIR / f"{pmid}.npy"
            if emb_file.exists():
                data['embedding'] = np.load(emb_file)
            else:
                data['embedding'] = None

        return data

    def set(self, pmid: str, abstract: str, chunks: List[str],
            embedding: Optional[np.ndarray] = None):
        """
        缓存文献数据

        Args:
            pmid: PubMed ID
            abstract: 摘要文本
            chunks: 切块列表
            embedding: 向量（可选）
        """
        cache_data = {
            'pmid': pmid,
            'abstract': abstract,
            'chunks': chunks,
            'has_embedding': embedding is not None,
            'chunk_count': len(chunks),
            'cached_at': datetime.now().isoformat(),
            'access_count': 0
        }

        self.cache[pmid] = cache_data

        # 保存embedding到单独文件
        if embedding is not None:
            emb_file = EMBEDDING_CACHE_DIR / f"{pmid}.npy"
            np.save(emb_file, embedding)

        # 保存索引
        self._save_cache()

    def get_cached_chunks(self, pmids: List[str]) -> Tuple[List[Dict], List[str]]:
        """
        获取已缓存的chunks和未缓存的PMID列表

        Returns:
            (cached_chunks_list, uncached_pmids_list)
        """
        cached_chunks = []
        uncached_pmids = []

        for pmid in pmids:
            cached_data = self.get(pmid)
            if cached_data:
                # 添加缓存数据到结果
                chunks = cached_data['chunks']

                # 检查chunks是否为空，如果为空则重新处理
                if not chunks:
                    print(f"    [Cache] PMID {pmid} has abstract but no chunks, will reprocess")
                    uncached_pmids.append(pmid)
                    continue

                # 将缓存的embedding按chunk挂到metadata，供下游合并（避免重复调用embedding API）
                embedding_matrix = cached_data.get('embedding')
                for chunk_idx, chunk_text in enumerate(chunks):
                    metadata = {
                        'chunk_id': f"{pmid}.{chunk_idx}",
                        'pmid': pmid,
                        'source': 'cached'
                    }
                    if embedding_matrix is not None and chunk_idx < len(embedding_matrix):
                        metadata['embedding'] = embedding_matrix[chunk_idx]
                    cached_chunks.append({
                        'text': chunk_text,
                        'metadata': metadata
                    })
            else:
                uncached_pmids.append(pmid)

        return cached_chunks, uncached_pmids

    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        total_pmids = len(self.cache)
        total_chunks = sum(data.get('chunk_count', 0) for data in self.cache.values())
        total_access = sum(data.get('access_count', 0) for data in self.cache.values())

        cache_size = 0
        if CACHE_FILE.exists():
            cache_size = CACHE_FILE.stat().st_size

        return {
            'total_pmids': total_pmids,
            'total_chunks': total_chunks,
            'total_access': total_access,
            'cache_size_bytes': cache_size
        }

    def clear(self):
        """清空缓存"""
        self.cache = {}
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        for file in EMBEDDING_CACHE_DIR.glob("*.npy"):
            file.unlink()
        print("[Cache] Cleared all cache")


_shared_pmid_cache: Optional[PMIDCache] = None


def get_shared_pmid_cache() -> PMIDCache:
    """进程级单例：避免每对实体重新从磁盘加载整个缓存JSON"""
    global _shared_pmid_cache
    if _shared_pmid_cache is None:
        _shared_pmid_cache = PMIDCache()
    return _shared_pmid_cache


# =============================================================================
# STEP 1: MeSH Term Lookup (Entity Normalization moved to lightrag_core.py)
# =============================================================================

class MeSHFinder:
    """MeSH术语查找器"""

    @staticmethod
    def expand_common_abbreviations(term: str) -> str:
        """展开常见的医学缩写

        Args:
            term: 原始术语

        Returns:
            展开后的术语
        """
        # 常见凝血因子缩写
        factor_map = {
            r'\bFV\b': 'Factor V',
            r'\bF5\b': 'Factor V',
            r'\bFVII\b': 'Factor VII',
            r'\bF7\b': 'Factor VII',
            r'\bFVIII\b': 'Factor VIII',
            r'\bF8\b': 'Factor VIII',
            r'\bFIX\b': 'Factor IX',
            r'\bF9\b': 'Factor IX',
            r'\bFX\b': 'Factor X',
            r'\bF10\b': 'Factor X',
        }

        expanded = term
        for pattern, full_name in factor_map.items():
            expanded = re.sub(pattern, full_name, expanded, flags=re.IGNORECASE)

        return expanded

    @staticmethod
    def extract_core_term(term: str) -> str:
        """从复杂描述中提取核心术语

        例如：
        - "Exacerbation of atrial fibrillation" -> "Atrial Fibrillation"
        - "History of myocardial infarction" -> "Myocardial Infarction"
        - "Chronic kidney disease" -> "Chronic Kidney Disease"

        Args:
            term: 原始术语

        Returns:
            提取的核心术语
        """
        # 常见的前缀模式（只移除真正不影响核心概念的修饰词）
        prefix_patterns = [
            r'^Exacerbation of\s+',
            r'^History of\s+',
            r'^Family history of\s+',
            r'^Risk of\s+',
            r'^Complication of\s+'
        ]

        cleaned = term
        for pattern in prefix_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # 如果清理后为空或太短，返回原始术语
        if not cleaned or len(cleaned.strip()) < 3:
            return term

        # 首字母大写（符合MeSH术语命名习惯）
        cleaned = ' '.join(word.capitalize() for word in cleaned.split())

        return cleaned

    _mesh_cache: Optional[Dict] = None

    @staticmethod
    def _load_mesh_cache() -> Dict:
        """加载MeSH查询缓存（进程内共享，首次访问时读盘）"""
        if MeSHFinder._mesh_cache is None:
            if ENTITY_CACHE_FILE.exists():
                try:
                    with open(ENTITY_CACHE_FILE, 'r', encoding='utf-8') as f:
                        MeSHFinder._mesh_cache = json.load(f)
                    print(f"[MeSH Cache] Loaded {len(MeSHFinder._mesh_cache)} cached terms")
                except Exception as e:
                    print(f"[MeSH Cache] Failed to load: {e}")
                    MeSHFinder._mesh_cache = {}
            else:
                MeSHFinder._mesh_cache = {}
        return MeSHFinder._mesh_cache

    @staticmethod
    def _store_mesh_cache(term: str, result: Optional[Dict]):
        """写入并落盘MeSH查询缓存"""
        cache = MeSHFinder._load_mesh_cache()
        if cache.get(term) == result:
            return
        cache[term] = result
        try:
            with open(ENTITY_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[MeSH Cache] Failed to save: {e}")

    @staticmethod
    def search_mesh(term: str, omop_retriever=None) -> Optional[Dict]:
        """通过NCBI API获取MeSH ID和名称

        Returns:
            Dict with 'id' (MeSH ID like D005955) and 'name', or None
        """
        # 步骤0: 展开常见医学缩写
        expanded_term = MeSHFinder.expand_common_abbreviations(term)
        if expanded_term != term:
            print(f"    [MeSH] 缩写展开: '{term}' -> '{expanded_term}'")
            term = expanded_term

        # 步骤1: 从复杂描述中提取核心术语
        core_term = MeSHFinder.extract_core_term(term)
        if core_term != term:
            print(f"    [MeSH] 提取核心术语: '{term}' -> '{core_term}'")
            term = core_term

        # 查询结果缓存（MeSH词条不变，命中即免NCBI请求）
        mesh_cache = MeSHFinder._load_mesh_cache()
        if term in mesh_cache:
            cached = mesh_cache[term]
            print(f"    [MeSH] 缓存命中: '{term}' -> {cached['id'] if cached else 'None'}")
            return dict(cached) if cached else None

        print(f"    [MeSH] 使用NCBI API搜索: {term}")
        url = f"{NCBI_BASE_URL}/esearch.fcgi"

        # 尝试多种搜索策略（从严格到宽松）
        search_strategies = [
            f'"{term}"[MeSH Terms]',           # 精确匹配MeSH术语
            f'"{term}"[MeSH NoExp]',           # MeSH NoExplode词汇
            f'{term}[MeSH Terms]',             # 部分匹配MeSH术语
            f'{term}[All Fields]',             # 所有字段
        ]

        for i, strategy in enumerate(search_strategies):
            params = {
                'db': 'mesh',
                'term': strategy,
                'retmax': 1,
                'tool': NCBI_TOOL,
                'email': NCBI_EMAIL
            }

            if NCBI_API_KEY:
                params['api_key'] = NCBI_API_KEY

            try:
                response = requests.get(url, params=params, proxies=PROXIES, timeout=10)
                response.raise_for_status()

                root = ET.fromstring(response.content)
                id_list = root.find('IdList')

                if id_list is not None and len(id_list) > 0:
                    mesh_id = id_list[0].text
                    print(f"    [MeSH] ✓ 找到匹配 (策略{i+1}: {strategy[:40]}...)")
                    print(f"    [MeSH] MeSH ID: {mesh_id}")

                    # 获取MeSH名称
                    mesh_name = MeSHFinder.get_mesh_term(mesh_id)
                    if mesh_name:
                        print(f"    [MeSH] MeSH名称: {mesh_name}")
                        result = {'id': mesh_id, 'name': mesh_name, 'cui': None}
                    else:
                        # 如果获取名称失败，但找到ID，返回使用原始术语名称
                        print(f"    [MeSH] 无法获取标准名称，使用原始术语: {term}")
                        result = {'id': mesh_id, 'name': term, 'cui': None}
                    MeSHFinder._store_mesh_cache(term, result)
                    return result

            except Exception as e:
                print(f"    [MeSH] 策略{i+1}失败: {e}")
                continue

        print(f"    [MeSH] 所有搜索策略均未找到结果")
        MeSHFinder._store_mesh_cache(term, None)
        return None

    @staticmethod
    def get_mesh_term(mesh_id: str) -> Optional[str]:
        """获取MeSH术语详情（支持标准MeSH和Supplementary Concept）"""
        url = f"{NCBI_BASE_URL}/esummary.fcgi"

        params = {
            'db': 'mesh',
            'id': mesh_id,
            'tool': NCBI_TOOL,
            'email': NCBI_EMAIL
        }

        if NCBI_API_KEY:
            params['api_key'] = NCBI_API_KEY

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            root = ET.fromstring(response.content)

            for doc_sum in root.findall('DocSum'):
                # 尝试多种字段名（标准MeSH和Supplementary Concept可能不同）
                term_names = ['MeshHeading', 'DescriptorName', 'Name', 'Title']

                for item in doc_sum.findall('Item'):
                    item_name = item.get('Name')

                    # 标准MeSH结构：MeshHeading -> DescriptorName
                    if item_name == 'MeshHeading':
                        for sub_item in item.findall('Item'):
                            if sub_item.get('Name') == 'DescriptorName':
                                term = sub_item.get('TermName') or sub_item.text
                                if term:
                                    return term.strip()

                    # Supplementary Concept结构：直接是Name
                    elif item_name in term_names:
                        term = item.get('TermName') or item.text
                        if term:
                            return term.strip()

                    # 尝试查找任何包含术语名称的Item
                    elif item_name and any(tn.lower() in item_name.lower() for tn in term_names):
                        term = item.get('TermName') or item.text
                        if term:
                            return term.strip()

            print(f"    [MeSH] 警告：无法从响应中提取术语名称 (ID: {mesh_id})")
            return None

        except Exception as e:
            print(f"    [MeSH] Get details failed: {e}")
            return None

    @staticmethod
    def find_mesh_terms(entity1: str, entity2: str, omop_retriever=None, original_entity1: str = None, original_entity2: str = None) -> Tuple[str, str, Dict, Dict]:
        """查找两个实体的MeSH术语

        Args:
            entity1: 第一个实体名称
            entity2: 第二个实体名称
            omop_retriever: 保留参数（未使用）
            original_entity1: 第一个实体的原始输入名称（用于MeSH匹配失败时的回退）
            original_entity2: 第二个实体的原始输入名称（用于MeSH匹配失败时的回退）

        Returns:
            Tuple of (search_term1, search_term2, result1_dict, result2_dict)
            - search_term: 用于PubMed搜索的概念名称
            - result_dict: 包含MeSH ID和概念的完整信息（用于图谱中的ISA匹配）
        """
        print(f"[MeSH Lookup]")

        # 使用原始实体名称作为回退（如果提供）
        fallback1 = original_entity1 if original_entity1 else entity1
        fallback2 = original_entity2 if original_entity2 else entity2

        result1 = MeSHFinder.search_mesh(entity1, omop_retriever)
        result2 = MeSHFinder.search_mesh(entity2, omop_retriever)

        # PubMed搜索始终使用概念名称（不是MeSH ID）
        # 因为Supplementary Concepts的MeSH ID不能用于PubMed检索
        if result1:
            term1 = result1['name']
            mesh_id1 = result1.get('id')
            if mesh_id1:
                print(f"  [{entity1}] -> {result1['name']} (MeSH: {mesh_id1})")
            else:
                print(f"  [{entity1}] -> {result1['name']} (无MeSH ID)")
        else:
            term1 = fallback1
            result1 = {}
            if fallback1 != entity1:
                print(f"  [{entity1}] 未找到，使用原始名称: '{fallback1}'")
            else:
                print(f"  [{entity1}] 未找到，使用原始名称")

        if result2:
            term2 = result2['name']
            mesh_id2 = result2.get('id')
            if mesh_id2:
                print(f"  [{entity2}] -> {result2['name']} (MeSH: {mesh_id2})")
            else:
                print(f"  [{entity2}] -> {result2['name']} (无MeSH ID)")
        else:
            term2 = fallback2
            result2 = {}
            if fallback2 != entity2:
                print(f"  [{entity2}] 未找到，使用原始名称: '{fallback2}'")
            else:
                print(f"  [{entity2}] 未找到，使用原始名称")

        print()

        return term1, term2, result1, result2


# =============================================================================
# STEP 3: PMID Search
# =============================================================================

class PMIDFinder:
    """PMID查找器"""

    @staticmethod
    def search_pubmed(query: str, max_results: int = 10) -> List[str]:
        """在PubMed中搜索"""
        url = f"{NCBI_BASE_URL}/esearch.fcgi"

        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'sort': 'relevance',  # 使用 Best Match (相关性) 排序
            'tool': NCBI_TOOL,
            'email': NCBI_EMAIL,
            'retmode': 'xml'
        }

        if NCBI_API_KEY:
            params['api_key'] = NCBI_API_KEY

        try:
            print(f"    [PMID] 查询: {query[:100]}...")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()

            root = ET.fromstring(response.content)

            pmids = []
            id_list = root.find('IdList')
            if id_list is not None:
                for id_elem in id_list.findall('Id'):
                    pmids.append(id_elem.text)
            else:
                print(f"    [PMID] 未找到IdList")
                # 调试：显示原始响应
                if len(response.content) < 500:
                    print(f"    [PMID] 原始响应: {response.content.decode('utf-8')[:200]}")

            print(f"    [PMID] 返回 {len(pmids)} 个结果")
            return pmids

        except Exception as e:
            print(f"    [PMID] Search failed: {type(e).__name__}: {e}")
            return []

    @staticmethod
    def find_pmids(term1: str, term2: str, entity1: str, entity2: str,
                   mesh_id1: str = None, mesh_id2: str = None, max_pmids: int = 10) -> List[str]:
        """查找PMID

        Args:
            term1: 第一个概念名称（MeSH Term或原始实体名称）
            term2: 第二个概念名称（MeSH Term或原始实体名称）
            entity1: 第一个原始实体名称
            entity2: 第二个原始实体名称
            mesh_id1: 第一个实体的MeSH ID（如果有）
            mesh_id2: 第二个实体的MeSH ID（如果有）

        Note:
            - 如果有MeSH ID，使用[MeSH Terms]查询
            - 如果没有MeSH ID，使用[Word]查询
        """
        print(f"[PMID Search]")

        # 构造查询：根据是否有MeSh ID选择标签
        def make_query_term(term: str, mesh_id: str) -> str:
            if mesh_id:
                # 有MeSh ID，使用[MeSH Term]查询
                return f'"{term}"[MeSH Term]'
            else:
                # 无MeSh ID，使用[Word]查询
                return f'"{term}"[Word]'

        # 组合搜索
        query1 = make_query_term(term1, mesh_id1)
        query2 = make_query_term(term2, mesh_id2)
        combined_query = f'({query1}) AND ({query2})'
        pmids = PMIDFinder.search_pubmed(combined_query, max_results=max_pmids)

        # 如果找到同时包含两个实体的文章，返回结果
        if pmids:
            print(f"  [Combined] Found {len(pmids)} PMIDs")
            print(f"[Result] {pmids[:max_pmids]}")
            print()
            return pmids[:max_pmids]

        # 如果没有文章同时包含两个实体，分别搜索
        print(f"  [Combined] No results, trying separate")

        # 调整：分别搜索时严格控制数量
        pmids1 = PMIDFinder.search_pubmed(query1, max_results=max_pmids//2)
        pmids2 = PMIDFinder.search_pubmed(query2, max_results=max_pmids//2)

        # 组合结果并去重（保持顺序）
        seen = set()
        all_pmids = []
        for pmid in pmids + pmids1 + pmids2:
            if pmid not in seen:
                seen.add(pmid)
                all_pmids.append(pmid)
                if len(all_pmids) >= max_pmids:
                    break

        print(f"  [Total] Found {len(all_pmids)} PMIDs (limited to {max_pmids})")
        print(f"[Result] {all_pmids}")
        print()

        return all_pmids


# =============================================================================
# STEP 4-6: Abstract Retrieval, Chunking, Embedding
# =============================================================================

class AbstractRetriever:
    """摘要提取器（支持缓存）"""

    @staticmethod
    def get_abstracts(pmids: List[str], cache: Optional[PMIDCache] = None) -> Tuple[List[Dict], List[str]]:
        """
        批量获取摘要（使用缓存）

        Args:
            pmids: PMID列表
            cache: 缓存实例（可选）

        Returns:
            (documents_list, uncached_pmids_list)
        """
        print(f"[Abstract] Fetching {len(pmids)} PMIDs")

        # 如果有缓存，先检查
        if cache:
            cached_pmids = []
            uncached_pmids = []

            for pmid in pmids:
                if pmid in cache.cache:
                    # 只检查是否在缓存中，不检查chunks
                    # chunks的检查在get_cached_chunks中进行
                    cached_pmids.append(pmid)
                else:
                    uncached_pmids.append(pmid)

            print(f"  Cache: {len(cached_pmids)} hits, {len(uncached_pmids)} misses")

            # 从缓存获取已处理的文档
            cached_documents = []
            for pmid in cached_pmids:
                cached_data = cache.get(pmid)
                if cached_data:
                    cached_documents.append({
                        'pmid': pmid,
                        'abstract': cached_data['abstract'],
                        'from_cache': True
                    })

            # 只获取未缓存的摘要
            if uncached_pmids:
                fresh_documents = AbstractRetriever._fetch_abstracts_from_pubmed(uncached_pmids)

                # 暂时不更新缓存（等切块后再更新）
                # 这样可以避免缓存chunks为空的情况

                all_documents = cached_documents + fresh_documents
            else:
                all_documents = cached_documents

            print(f"[Result] {len(all_documents)} abstracts ({len(cached_documents)} from cache)")

            return all_documents, uncached_pmids
        else:
            # 无缓存，直接获取
            documents = AbstractRetriever._fetch_abstracts_from_pubmed(pmids)
            print(f"[Result] {len(documents)} abstracts")
            print()
            return documents, pmids

    @staticmethod
    def _fetch_abstracts_from_pubmed(pmids: List[str]) -> List[Dict]:
        """从PubMed获取摘要（内部方法）"""
        documents = []

        for i in range(0, len(pmids), 100):
            batch = pmids[i:i+100]
            url = f"{NCBI_BASE_URL}/efetch.fcgi"

            params = {
                'db': 'pubmed',
                'id': ','.join(batch),
                'retmode': 'xml',
                'tool': NCBI_TOOL,
                'email': NCBI_EMAIL
            }

            if NCBI_API_KEY:
                params['api_key'] = NCBI_API_KEY

            try:
                response = requests.get(url, params=params, proxies=PROXIES, timeout=20)
                response.raise_for_status()

                root = ET.fromstring(response.content)

                for article in root.findall('.//PubmedArticle'):
                    pmid_elem = article.find('.//PMID')

                    if pmid_elem is not None:
                        pmid = pmid_elem.text
                        abstract = ""

                        # 拼接全部AbstractText段（结构化摘要含BACKGROUND/METHODS/RESULTS/CONCLUSIONS等多段；
                        # itertext递归取全文本，避免<i>/<b>等内联标签导致文字截断）
                        for text in article.findall('.//AbstractText'):
                            content = ''.join(text.itertext()).strip()
                            if content:
                                label = text.get('Label', '')
                                if label:
                                    abstract += f"{label}: {content} "
                                else:
                                    abstract += content + " "
                        abstract = abstract.strip()

                        if abstract:
                            documents.append({'pmid': pmid, 'abstract': abstract, 'from_cache': False})

                time.sleep(0.34)

            except Exception as e:
                print(f"    [Abstract] Batch failed: {e}")

        return documents


import nltk

class TextChunker:
    """文本切块器"""

    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt_tab/english/')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
            try:
                nltk.data.find('tokenizers/punkt_tab/english/')
            except LookupError as e:
                raise RuntimeError(
                    "NLTK分句资源 punkt_tab 不可用，请手动执行: python -c \"import nltk; nltk.download('punkt_tab')\""
                ) from e

    def chunk(self, text: str) -> List[str]:
        """切块"""
        sentences = nltk.sent_tokenize(text)
        chunk1 = [s.strip() for s in sentences if s.strip()]

        all_chunks_doc3 = []
        if len(chunk1) > 3:
            for i in range(len(chunk1) - 2):
                all_chunks_doc3.append(" ".join(chunk1[i:i+3]))

        return chunk1 + all_chunks_doc3


async def get_qwen_embeddings(texts: List[str]) -> np.ndarray:
    """获取Qwen embeddings"""
    client = AsyncOpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

    batch_size = 10
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = await client.embeddings.create(
            model=QWEN_EMBEDDING_MODEL,
            input=batch,
            encoding_format="float"
        )
        all_embeddings.extend([item.embedding for item in response.data])

        if (i // batch_size + 1) % 5 == 0:
            print(f"    Embedded {len(all_embeddings)}/{len(texts)}...")

    return np.array(all_embeddings, dtype=np.float32)


from numpy.linalg import norm

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算余弦相似度"""
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


class ChunkRetriever:
    """切块检索器（支持Rerank精排）"""

    def __init__(self, enable_rerank: bool = True):
        self.chunks = []
        self.embeddings = None
        self.chunk_metadata = []
        self.enable_rerank = enable_rerank
        self.rerank_func = None

        # 延迟初始化rerank函数
        if enable_rerank:
            try:
                from lightrag.rerank import create_aliyun_rerank_func

                # 使用通义千问 API Key
                api_key = os.getenv("DASHSCOPE_API_KEY", "")
                if not api_key:
                    print("[ChunkRetriever] 警告：DASHSCOPE_API_KEY 未设置，Rerank功能已禁用")
                    self.enable_rerank = False
                else:
                    self.rerank_func = create_aliyun_rerank_func(
                        model="gte-rerank-v2",
                        top_n=20,  # 召回数量
                        api_key=api_key
                    )
                    print(f"[ChunkRetriever] Rerank功能已启用（通义千问 API: gte-rerank-v2）")
            except Exception as e:
                print(f"[ChunkRetriever] Rerank初始化失败: {e}，将使用向量相似度")
                self.enable_rerank = False

    def load_chunks(self, chunks_data: List[Dict], embeddings: np.ndarray):
        """加载切块数据"""
        self.chunks = [c['text'] for c in chunks_data]
        self.embeddings = embeddings
        self.chunk_metadata = [c['metadata'] for c in chunks_data]

    async def retrieve(self, query: str, top_k: int = 3, recall_k: int = 20) -> List[Dict]:
        """检索最相关chunks（支持Rerank精排）

        Args:
            query: 查询文本
            top_k: 最终返回数量
            recall_k: 向量检索召回数量（用于rerank）

        Returns:
            检索结果列表，每个结果包含 chunk_text, similarity, rerank_score, combined_score, metadata
        """
        # === 阶段1: 向量相似度检索（召回） ===
        query_emb = await get_qwen_embeddings([query])
        query_vec = query_emb[0]

        similarities = []
        for idx, chunk_emb in enumerate(self.embeddings):
            sim = cosine_similarity(query_vec, chunk_emb)
            similarities.append({
                'idx': idx,
                'chunk_text': self.chunks[idx],
                'similarity': float(sim),
                'metadata': self.chunk_metadata[idx]
            })

        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        # 召回更多候选（用于rerank）
        recall_chunks = similarities[:recall_k] if len(similarities) > recall_k else similarities

        if not self.enable_rerank or not self.rerank_func or len(recall_chunks) <= top_k:
            # 不使用rerank或候选数量不足，直接返回
            print(f"    [ChunkRetriever] 向量检索: {len(recall_chunks)} chunks → top-{top_k}")
            return similarities[:top_k]

        # === 阶段2: Rerank精排 ===
        try:
            print(f"    [ChunkRetriever] Rerank精排: {len(recall_chunks)} chunks → top-{top_k}")

            # 准备文档列表
            documents = [item['chunk_text'][:1500] for item in recall_chunks]  # 限制长度

            # 调用rerank
            rerank_results = await self.rerank_func(query, documents)

            # 合并rerank分数
            rerank_scores = {}
            for result in rerank_results:
                orig_idx = result['index']
                score = result['relevance_score']
                if 0 <= orig_idx < len(recall_chunks):
                    rerank_scores[orig_idx] = score

            # 计算混合分数：70% rerank + 30% 向量相似度
            for idx, item in enumerate(recall_chunks):
                rerank_score = rerank_scores.get(idx, 0.0)
                item['rerank_score'] = rerank_score
                item['combined_score'] = 0.7 * rerank_score + 0.3 * item['similarity']

            # 按混合分数重新排序
            recall_chunks.sort(key=lambda x: x['combined_score'], reverse=True)

            # 显示rerank效果
            print(f"    [ChunkRetriever] Rerank完成，Top-{top_k} 结果:")
            for i, item in enumerate(recall_chunks[:top_k]):
                vec_sim = item['similarity']
                rerank = item.get('rerank_score', 0.0)
                combined = item.get('combined_score', 0.0)
                pmid = item['metadata'].get('pmid', 'N/A')
                print(f"      [{i+1}] PMID:{pmid} | 向量:{vec_sim:.3f} | Rerank:{rerank:.3f} | 混合:{combined:.3f}")

            return recall_chunks[:top_k]

        except Exception as e:
            print(f"    [ChunkRetriever] Rerank失败: {e}，回退到向量相似度")
            return similarities[:top_k]


# =============================================================================
# Main Pipeline
# =============================================================================

class EnhancedDynamicRAG:
    """增强的动态RAG管道（支持Rerank精排）"""

    def __init__(self, max_pmids: int = 10, use_cache: bool = True, enable_rerank: bool = True):
        self.max_pmids = max_pmids
        self.use_cache = use_cache
        self.omop_retriever = None

        self.chunker = TextChunker()
        self.retriever = ChunkRetriever(enable_rerank=enable_rerank)

        # 初始化缓存
        if self.use_cache:
            self.cache = get_shared_pmid_cache()
            stats = self.cache.get_stats()
            print(f"[Cache] Initialized with {stats['total_pmids']} PMIDs, "
                  f"{stats['total_chunks']} chunks")
        else:
            self.cache = None

    async def build_rag_for_query(self, entity1: str, entity2: str,
                                  entity1_type: str = "", entity2_type: str = "",
                                  abstract: str = "") -> Dict:
        """为查询构建动态RAG库（使用缓存）"""
        print("=" * 70)
        print("ENHANCED DYNAMIC RAG")
        print("=" * 70)
        print(f"Query: {entity1} <-> {entity2}")
        print()

        # 显示缓存统计
        if self.cache:
            stats = self.cache.get_stats()
            print(f"[Cache Stats] PMIDs: {stats['total_pmids']}, "
                  f"Chunks: {stats['total_chunks']}, "
                  f"Access: {stats['total_access']}")
            print()

        # Step 1: 直接使用原始实体名称
        print(f"[Entity Normalization] Using original entity names")
        print(f"[Result] Using entities: '{entity1}' | '{entity2}'")
        print()

        # Step 2: MeSH查找
        search_term1, search_term2, mesh_result1, mesh_result2 = MeSHFinder.find_mesh_terms(
            entity1, entity2, omop_retriever=None,
            original_entity1=entity1, original_entity2=entity2
        )

        # Step 3: PMID查找（传递MeSh ID用于选择查询类型）
        mesh_id1 = mesh_result1.get('id') if mesh_result1 else None
        mesh_id2 = mesh_result2.get('id') if mesh_result2 else None
        pmids = PMIDFinder.find_pmids(search_term1, search_term2, entity1, entity2,
                                       mesh_id1=mesh_id1, mesh_id2=mesh_id2,
                                       max_pmids=self.max_pmids)

        if not pmids:
            return {
                'success': False,
                'message': 'No PMIDs found',
                'normalized_entities': (entity1, entity2),
                'mesh_terms': (search_term1, search_term2)
            }

        # Step 4: 使用缓存获取摘要和chunks
        if self.cache:
            cached_chunks, uncached_pmids = self.cache.get_cached_chunks(pmids)
            print(f"[Cache] {len(cached_chunks)} cached chunks, "
                  f"{len(uncached_pmids)} PMIDs to fetch")
        else:
            cached_chunks = []
            uncached_pmids = pmids

        # 获取未缓存的摘要
        if uncached_pmids:
            documents, _ = AbstractRetriever.get_abstracts(uncached_pmids, cache=self.cache)
        else:
            documents = []

        # 从缓存中恢复文档（用于临时图谱构建）
        cached_documents = []
        if self.cache and cached_chunks:
            for pmid in pmids:
                # 跳过已在新documents中的PMID
                if any(doc.get('pmid') == pmid for doc in documents):
                    continue
                cached_data = self.cache.get(pmid)
                if cached_data and cached_data.get('abstract'):
                    cached_documents.append({
                        'pmid': pmid,
                        'abstract': cached_data['abstract'],
                        'from_cache': True
                    })

        # 合并documents和cached_documents
        all_documents = documents + cached_documents

        # Step 5: 处理未缓存的文档 -> 切块 -> 向量化 -> 缓存
        fresh_chunks = []
        fresh_metadata = []

        if documents:
            print(f"[Processing] {len(documents)} new documents")

            for doc in documents:
                if doc.get('from_cache'):
                    # 从缓存获取（已有摘要但需要切块）
                    pmid = doc['pmid']
                    cached_data = self.cache.get(pmid)
                    if cached_data and cached_data.get('chunks'):
                        # 缓存中有chunks，使用缓存的chunks
                        chunks = cached_data['chunks']
                        for chunk_idx, chunk_text in enumerate(chunks):
                            fresh_chunks.append(chunk_text)
                            fresh_metadata.append({
                                'chunk_id': f"{pmid}.{chunk_idx}",
                                'pmid': pmid,
                                'source': 'cached'
                            })
                        continue
                    else:
                        # 缓存中无chunks，重新切块
                        print(f"    [Processing] PMID {pmid} from cache but no chunks, re-chunking")
                        chunks = self.chunker.chunk(doc['abstract'])
                        for chunk_idx, chunk_text in enumerate(chunks):
                            fresh_chunks.append(chunk_text)
                            fresh_metadata.append({
                                'chunk_id': f"{pmid}.{chunk_idx}",
                                'pmid': pmid,
                                'source': 'recached'
                            })
                        continue

                # 新文档：切块
                chunks = self.chunker.chunk(doc['abstract'])
                pmid = doc.get('pmid', 'unknown')

                for chunk_idx, chunk_text in enumerate(chunks):
                    fresh_chunks.append(chunk_text)
                    fresh_metadata.append({
                        'chunk_id': f"{pmid}.{chunk_idx}",
                        'pmid': pmid,
                        'source': 'dynamic_pubmed'
                    })

            # 向量化新chunks
            if fresh_chunks:
                print(f"[Embedding] {len(fresh_chunks)} new chunks")
                fresh_embeddings = await get_qwen_embeddings(fresh_chunks)

                # 缓存新处理的PMID
                for doc in documents:
                    pmid = doc['pmid']
                    doc_chunks = [c for c, m in zip(fresh_chunks, fresh_metadata)
                                 if m['pmid'] == pmid]
                    doc_embeddings = [e for c, e, m in
                                      zip(fresh_chunks, fresh_embeddings, fresh_metadata)
                                      if m['pmid'] == pmid]
                    if doc_chunks and doc_embeddings is not None:
                        self.cache.set(pmid, doc['abstract'], doc_chunks,
                                     np.array(doc_embeddings))
                    else:
                        # 即使没有chunks，也缓存摘要
                        self.cache.set(pmid, doc['abstract'], [])

                # 合并cached chunks的embedding
                if cached_chunks and 'embedding' in cached_chunks[0].get('metadata', {}):
                    cached_embeddings = [c['metadata']['embedding'] for c in cached_chunks]
                    all_embeddings = np.vstack([cached_embeddings, fresh_embeddings])
                else:
                    all_embeddings = fresh_embeddings

                all_chunks = [c['text'] if isinstance(c, dict) else c
                              for c in cached_chunks] + fresh_chunks
                all_metadata = [c.get('metadata', c) if isinstance(c, dict) else {'source': 'cached'}
                                for c in cached_chunks] + fresh_metadata
            else:
                # 只有cached chunks
                all_chunks = [c['text'] if isinstance(c, dict) else c for c in cached_chunks]
                all_metadata = [c.get('metadata', c) if isinstance(c, dict) else c
                                for c in cached_chunks]
                all_embeddings = None
        else:
            # 只有cached chunks
            all_chunks = [c['text'] if isinstance(c, dict) else c for c in cached_chunks]
            all_metadata = [c.get('metadata', c) if isinstance(c, dict) else c
                            for c in cached_chunks]
            all_embeddings = None

        print(f"[Total] {len(all_chunks)} chunks ready")

        # 全部chunks命中缓存时，从缓存恢复embeddings（避免重复调用embedding API）
        if (all_embeddings is None or (isinstance(all_embeddings, np.ndarray) and
                                       all_embeddings.size == 0)) and cached_chunks and all_chunks:
            cached_embeddings = [c.get('metadata', {}).get('embedding') for c in cached_chunks
                                 if isinstance(c, dict)]
            if (len(cached_embeddings) == len(all_chunks)
                    and all(e is not None for e in cached_embeddings)):
                all_embeddings = np.vstack(cached_embeddings)
                print(f"[Embedding] 从缓存恢复 {len(cached_embeddings)} 个chunk向量（跳过API调用）")

        # 如果没有embeddings（全部来自缓存且未保存），重新生成
        if all_embeddings is None or (isinstance(all_embeddings, np.ndarray) and
                                      all_embeddings.size == 0):
            print(f"[Embedding] Generating embeddings for {len(all_chunks)} chunks")
            all_embeddings = await get_qwen_embeddings(all_chunks)
        elif isinstance(all_embeddings, np.ndarray) and all_embeddings.shape[0] != len(all_chunks):
            print(f"[Embedding] Regenerating (size mismatch)")
            all_embeddings = await get_qwen_embeddings(all_chunks)

        # Step 6: Retrieval
        chunks_data = [
            {'text': t, 'metadata': m}
            for t, m in zip(all_chunks, all_metadata)
        ]
        self.retriever.load_chunks(chunks_data, all_embeddings)

        query_text = f"What is the relationship between {entity1} and {entity2}?"
        retrieval_results = await self.retriever.retrieve(query_text, top_k=5)

        print(f"[Retrieval] Retrieved {len(retrieval_results)} chunks")

        # 更新缓存统计
        if self.cache:
            stats = self.cache.get_stats()
            print(f"[Cache Updated] PMIDs: {stats['total_pmids']}, "
                  f"Chunks: {stats['total_chunks']}")

        # 判断是否使用了原始实体名称（用于图谱查询）
        use_original_entity1 = (search_term1 == entity1)
        use_original_entity2 = (search_term2 == entity2)

        return {
            'success': True,
            'original_entities': (entity1, entity2),
            'normalized_entities': (entity1, entity2),
            'mesh_terms': (search_term1, search_term2),
            'use_original_entities': (use_original_entity1, use_original_entity2),  # 添加：是否使用原始实体
            'pmids': pmids,
            'documents': all_documents,  # 包含缓存和新获取的文档
            'chunks': all_chunks,  # 添加：切块后的文本（用于图谱构建）
            'chunk_metadata': all_metadata,  # 添加：chunks的元数据
            'embeddings': all_embeddings,  # 添加：chunks的向量
            'total_chunks': len(all_chunks),
            'cached_chunks': len(cached_chunks),
            'fresh_chunks': len(fresh_chunks) if documents else 0,
            'retrieval_results': retrieval_results  # 保留用于调试和可视化
        }


# =============================================================================
# Demo
# =============================================================================

async def demo():
    """演示增强的动态RAG（带缓存效果展示）"""
    rag = EnhancedDynamicRAG(max_pmids=10)

    # 测试案例 - 重复查询以展示缓存效果
    test_cases = [
        ("HNF-6", "Type II diabetes", "GeneOrGeneProduct", "DiseaseOrPhenotypicFeature"),
        ("HNF-6", "Type II diabetes", "GeneOrGeneProduct", "DiseaseOrPhenotypicFeature"),  # 重复 - 应该命中缓存
        ("BRCA1", "breast cancer", "GeneOrGeneProduct", "DiseaseOrPhenotypicFeature"),
        ("insulin", "glucose", "", ""),
    ]

    for idx, (entity1, entity2, type1, type2) in enumerate(test_cases):
        is_repeat = idx > 0 and test_cases[idx] == test_cases[idx-1]
        repeat_note = " (REPEATED QUERY - Should use cache)" if is_repeat else ""

        print(f"\n{'='*70}")
        print(f"TEST {idx+1}: {entity1} ({type1}) <-> {entity2} ({type2}){repeat_note}")
        print(f"{'='*70}\n")

        result = await rag.build_rag_for_query(
            entity1, entity2, type1, type2
        )

        if result['success']:
            print(f"\n{'='*70}")
            print(f"RESULTS")
            print(f"{'='*70}")
            print(f"Original:     {result['original_entities']}")
            print(f"Normalized:   {result['normalized_entities']}")
            print(f"MeSH:         {result['mesh_terms']}")
            print(f"PMIDs:        {len(result['pmids'])} ({result['pmids'][:3]}...)")
            print(f"Documents:    {len(result['documents'])}")
            print(f"Total Chunks: {result['total_chunks']}")
            print(f"  - Cached:   {result.get('cached_chunks', 0)}")
            print(f"  - Fresh:     {result.get('fresh_chunks', 0)}")

            print(f"\n[TOP 3 CHUNKS]")
            for i, r in enumerate(result['retrieval_results'][:3]):
                print(f"  [{i+1}] Sim: {r['similarity']:.4f}")
                print(f"      PMID: {r['metadata']['pmid']}")
                print(f"      Source: {r['metadata'].get('source', 'N/A')}")
                text = r['chunk_text'][:80]
                print(f"      Text: {text}...")
        else:
            print(f"[FAILED] {result['message']}")
            print(f"  Normalized: {result.get('normalized_entities', 'N/A')}")
            print(f"  MeSH: {result.get('mesh_terms', 'N/A')}")

        print()

    # 最终缓存统计
    print("\n" + "=" * 70)
    print("FINAL CACHE STATISTICS")
    print("=" * 70)
    stats = rag.cache.get_stats()
    print(f"Total PMIDs cached: {stats['total_pmids']}")
    print(f"Total chunks cached: {stats['total_chunks']}")
    print(f"Total cache accesses: {stats['total_access']}")
    print(f"Cache size: {stats['cache_size_bytes'] / 1024:.2f} KB")
    print()


if __name__ == "__main__":
    if not QWEN_API_KEY:
        print("ERROR: DASHSCOPE_API_KEY not found")
        exit(1)

    print("Enhanced Dynamic RAG")
    print(f"Embedding: {QWEN_EMBEDDING_MODEL}\n")

    asyncio.run(demo())
