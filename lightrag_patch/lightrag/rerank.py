from __future__ import annotations

import os
import aiohttp
from typing import Any, List, Dict, Optional, Tuple
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from .utils import logger

from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


def chunk_documents_for_rerank(
    documents: List[str],
    max_tokens: int = 480,
    overlap_tokens: int = 32,
    tokenizer_model: str = "gpt-4o-mini",
) -> Tuple[List[str], List[int]]:
    """
    Chunk documents that exceed token limit for reranking.

    Args:
        documents: List of document strings to chunk
        max_tokens: Maximum tokens per chunk (default 480 to leave margin for 512 limit)
        overlap_tokens: Number of tokens to overlap between chunks
        tokenizer_model: Model name for tiktoken tokenizer

    Returns:
        Tuple of (chunked_documents, original_doc_indices)
        - chunked_documents: List of document chunks (may be more than input)
        - original_doc_indices: Maps each chunk back to its original document index
    """
    # Clamp overlap_tokens to ensure the loop always advances
    # If overlap_tokens >= max_tokens, the chunking loop would hang
    if overlap_tokens >= max_tokens:
        original_overlap = overlap_tokens
        # Ensure overlap is at least 1 token less than max to guarantee progress
        # For very small max_tokens (e.g., 1), set overlap to 0
        overlap_tokens = max(0, max_tokens - 1)
        logger.warning(
            f"overlap_tokens ({original_overlap}) must be less than max_tokens ({max_tokens}). "
            f"Clamping to {overlap_tokens} to prevent infinite loop."
        )

    try:
        from .utils import TiktokenTokenizer

        tokenizer = TiktokenTokenizer(model_name=tokenizer_model)
    except Exception as e:
        logger.warning(
            f"Failed to initialize tokenizer: {e}. Using character-based approximation."
        )
        # Fallback: approximate 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        overlap_chars = overlap_tokens * 4

        chunked_docs = []
        doc_indices = []

        for idx, doc in enumerate(documents):
            if len(doc) <= max_chars:
                chunked_docs.append(doc)
                doc_indices.append(idx)
            else:
                # Split into overlapping chunks
                start = 0
                while start < len(doc):
                    end = min(start + max_chars, len(doc))
                    chunk = doc[start:end]
                    chunked_docs.append(chunk)
                    doc_indices.append(idx)

                    if end >= len(doc):
                        break
                    start = end - overlap_chars

        return chunked_docs, doc_indices

    # Use tokenizer for accurate chunking
    chunked_docs = []
    doc_indices = []

    for idx, doc in enumerate(documents):
        tokens = tokenizer.encode(doc)

        if len(tokens) <= max_tokens:
            # Document fits in one chunk
            chunked_docs.append(doc)
            doc_indices.append(idx)
        else:
            # Split into overlapping chunks
            start = 0
            while start < len(tokens):
                end = min(start + max_tokens, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = tokenizer.decode(chunk_tokens)
                chunked_docs.append(chunk_text)
                doc_indices.append(idx)

                if end >= len(tokens):
                    break
                start = end - overlap_tokens

    return chunked_docs, doc_indices


def aggregate_chunk_scores(
    chunk_results: List[Dict[str, Any]],
    doc_indices: List[int],
    num_original_docs: int,
    aggregation: str = "max",
) -> List[Dict[str, Any]]:
    """
    Aggregate rerank scores from document chunks back to original documents.

    Args:
        chunk_results: Rerank results for chunks [{"index": chunk_idx, "relevance_score": score}, ...]
        doc_indices: Maps each chunk index to original document index
        num_original_docs: Total number of original documents
        aggregation: Strategy for aggregating scores ("max", "mean", "first")

    Returns:
        List of results for original documents [{"index": doc_idx, "relevance_score": score}, ...]
    """
    # Group scores by original document index
    doc_scores: Dict[int, List[float]] = {i: [] for i in range(num_original_docs)}

    for result in chunk_results:
        chunk_idx = result["index"]
        score = result["relevance_score"]

        if 0 <= chunk_idx < len(doc_indices):
            original_doc_idx = doc_indices[chunk_idx]
            doc_scores[original_doc_idx].append(score)

    # Aggregate scores
    aggregated_results = []
    for doc_idx, scores in doc_scores.items():
        if not scores:
            continue

        if aggregation == "max":
            final_score = max(scores)
        elif aggregation == "mean":
            final_score = sum(scores) / len(scores)
        elif aggregation == "first":
            final_score = scores[0]
        else:
            logger.warning(f"Unknown aggregation strategy: {aggregation}, using max")
            final_score = max(scores)

        aggregated_results.append(
            {
                "index": doc_idx,
                "relevance_score": final_score,
            }
        )

    # Sort by relevance score (descending)
    aggregated_results.sort(key=lambda x: x["relevance_score"], reverse=True)

    return aggregated_results


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(
        retry_if_exception_type(aiohttp.ClientError)
        | retry_if_exception_type(aiohttp.ClientResponseError)
    ),
)
async def generic_rerank_api(
    query: str,
    documents: List[str],
    model: str,
    base_url: str,
    api_key: Optional[str],
    top_n: Optional[int] = None,
    return_documents: Optional[bool] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    response_format: str = "standard",  # "standard" (Jina/Cohere) or "aliyun"
    request_format: str = "standard",  # "standard" (Jina/Cohere) or "aliyun"
    enable_chunking: bool = False,
    max_tokens_per_doc: int = 480,
) -> List[Dict[str, Any]]:
    """
    Generic rerank API call for Jina/Cohere/Aliyun models.

    Args:
        query: The search query
        documents: List of strings to rerank
        model: Model name to use
        base_url: API endpoint URL
        api_key: API key for authentication
        top_n: Number of top results to return
        return_documents: Whether to return document text (Jina only)
        extra_body: Additional body parameters
        response_format: Response format type ("standard" for Jina/Cohere, "aliyun" for Aliyun)
        request_format: Request format type
        enable_chunking: Whether to chunk documents exceeding token limit
        max_tokens_per_doc: Maximum tokens per document for chunking

    Returns:
        List of dictionary of ["index": int, "relevance_score": float]
    """
    if not base_url:
        raise ValueError("Base URL is required")

    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    # Handle document chunking if enabled
    original_documents = documents
    doc_indices = None
    original_top_n = top_n  # Save original top_n for post-aggregation limiting

    if enable_chunking:
        documents, doc_indices = chunk_documents_for_rerank(
            documents, max_tokens=max_tokens_per_doc
        )
        logger.debug(
            f"Chunked {len(original_documents)} documents into {len(documents)} chunks"
        )
        # When chunking is enabled, disable top_n at API level to get all chunk scores
        # This ensures proper document-level coverage after aggregation
        # We'll apply top_n to aggregated document results instead
        if top_n is not None:
            logger.debug(
                f"Chunking enabled: disabled API-level top_n={top_n} to ensure complete document coverage"
            )
            top_n = None

    # Build request payload based on request format
    if request_format == "aliyun":
        # Aliyun format: nested input/parameters structure
        payload = {
            "model": model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {},
        }

        # Add optional parameters to parameters object
        if top_n is not None:
            payload["parameters"]["top_n"] = top_n

        if return_documents is not None:
            payload["parameters"]["return_documents"] = return_documents

        # Add extra parameters to parameters object
        if extra_body:
            payload["parameters"].update(extra_body)
    else:
        # Standard format for Jina/Cohere/OpenAI
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
        }

        # Add optional parameters
        if top_n is not None:
            payload["top_n"] = top_n

        # Only Jina API supports return_documents parameter
        if return_documents is not None and response_format in ("standard",):
            payload["return_documents"] = return_documents

        # Add extra parameters
        if extra_body:
            payload.update(extra_body)

    logger.debug(
        f"Rerank request: {len(documents)} documents, model: {model}, format: {response_format}"
    )

    async with aiohttp.ClientSession() as session:
        async with session.post(base_url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                content_type = response.headers.get("content-type", "").lower()
                is_html_error = (
                    error_text.strip().startswith("<!DOCTYPE html>")
                    or "text/html" in content_type
                )
                if is_html_error:
                    if response.status == 502:
                        clean_error = "Bad Gateway (502) - Rerank service temporarily unavailable. Please try again in a few minutes."
                    elif response.status == 503:
                        clean_error = "Service Unavailable (503) - Rerank service is temporarily overloaded. Please try again later."
                    elif response.status == 504:
                        clean_error = "Gateway Timeout (504) - Rerank service request timed out. Please try again."
                    else:
                        clean_error = f"HTTP {response.status} - Rerank service error. Please try again later."
                else:
                    clean_error = error_text
                logger.error(f"Rerank API error {response.status}: {clean_error}")
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"Rerank API error: {clean_error}",
                )

            response_json = await response.json()

            if response_format == "aliyun":
                # Aliyun format: {"output": {"results": [...]}}
                results = response_json.get("output", {}).get("results", [])
                if not isinstance(results, list):
                    logger.warning(
                        f"Expected 'output.results' to be list, got {type(results)}: {results}"
                    )
                    results = []
            elif response_format == "standard":
                # Standard format: {"results": [...]}
                results = response_json.get("results", [])
                if not isinstance(results, list):
                    logger.warning(
                        f"Expected 'results' to be list, got {type(results)}: {results}"
                    )
                    results = []
            else:
                raise ValueError(f"Unsupported response format: {response_format}")

            if not results:
                logger.warning("Rerank API returned empty results")
                return []

            # Standardize return format
            standardized_results = [
                {"index": result["index"], "relevance_score": result["relevance_score"]}
                for result in results
            ]

            # Aggregate chunk scores back to original documents if chunking was enabled
            if enable_chunking and doc_indices:
                standardized_results = aggregate_chunk_scores(
                    standardized_results,
                    doc_indices,
                    len(original_documents),
                    aggregation="max",
                )
                # Apply original top_n limit at document level (post-aggregation)
                # This preserves document-level semantics: top_n limits documents, not chunks
                if (
                    original_top_n is not None
                    and len(standardized_results) > original_top_n
                ):
                    standardized_results = standardized_results[:original_top_n]

            return standardized_results


async def cohere_rerank(
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
    api_key: Optional[str] = None,
    model: str = "rerank-v3.5",
    base_url: str = "https://api.cohere.com/v2/rerank",
    extra_body: Optional[Dict[str, Any]] = None,
    enable_chunking: bool = False,
    max_tokens_per_doc: int = 4096,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Cohere API.

    Supports both standard Cohere API and Cohere-compatible proxies

    Args:
        query: The search query
        documents: List of strings to rerank
        top_n: Number of top results to return
        api_key: API key for authentication
        model: rerank model name (default: rerank-v3.5)
        base_url: API endpoint
        extra_body: Additional body for http request(reserved for extra params)
        enable_chunking: Whether to chunk documents exceeding max_tokens_per_doc
        max_tokens_per_doc: Maximum tokens per document (default: 4096 for Cohere v3.5)

    Returns:
        List of dictionary of ["index": int, "relevance_score": float]

    Example:
        >>> # Standard Cohere API
        >>> results = await cohere_rerank(
        ...     query="What is the meaning of life?",
        ...     documents=["Doc1", "Doc2"],
        ...     api_key="your-cohere-key"
        ... )

        >>> # LiteLLM proxy with user authentication
        >>> results = await cohere_rerank(
        ...     query="What is vector search?",
        ...     documents=["Doc1", "Doc2"],
        ...     model="answerai-colbert-small-v1",
        ...     base_url="https://llm-proxy.example.com/v2/rerank",
        ...     api_key="your-proxy-key",
        ...     enable_chunking=True,
        ...     max_tokens_per_doc=480
        ... )
    """
    if api_key is None:
        api_key = os.getenv("COHERE_API_KEY") or os.getenv("RERANK_BINDING_API_KEY")

    return await generic_rerank_api(
        query=query,
        documents=documents,
        model=model,
        base_url=base_url,
        api_key=api_key,
        top_n=top_n,
        return_documents=None,  # Cohere doesn't support this parameter
        extra_body=extra_body,
        response_format="standard",
        enable_chunking=enable_chunking,
        max_tokens_per_doc=max_tokens_per_doc,
    )


async def jina_rerank(
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
    api_key: Optional[str] = None,
    model: str = "jina-reranker-v2-base-multilingual",
    base_url: str = "https://api.jina.ai/v1/rerank",
    extra_body: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Jina AI API.

    Args:
        query: The search query
        documents: List of strings to rerank
        top_n: Number of top results to return
        api_key: API key
        model: rerank model name
        base_url: API endpoint
        extra_body: Additional body for http request(reserved for extra params)

    Returns:
        List of dictionary of ["index": int, "relevance_score": float]
    """
    if api_key is None:
        api_key = os.getenv("JINA_API_KEY") or os.getenv("RERANK_BINDING_API_KEY")

    return await generic_rerank_api(
        query=query,
        documents=documents,
        model=model,
        base_url=base_url,
        api_key=api_key,
        top_n=top_n,
        return_documents=False,
        extra_body=extra_body,
        response_format="standard",
    )


async def ali_rerank(
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
    api_key: Optional[str] = None,
    model: str = "gte-rerank-v2",
    base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    extra_body: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Aliyun DashScope API.

    Args:
        query: The search query
        documents: List of strings to rerank
        top_n: Number of top results to return
        api_key: Aliyun API key
        model: rerank model name
        base_url: API endpoint
        extra_body: Additional body for http request(reserved for extra params)

    Returns:
        List of dictionary of ["index": int, "relevance_score": float]
    """
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("RERANK_BINDING_API_KEY")

    return await generic_rerank_api(
        query=query,
        documents=documents,
        model=model,
        base_url=base_url,
        api_key=api_key,
        top_n=top_n,
        return_documents=False,  # Aliyun doesn't need this parameter
        extra_body=extra_body,
        response_format="aliyun",
        request_format="aliyun",
    )


def create_aliyun_rerank_func(
    model: str = "gte-rerank-v2",
    top_n: Optional[int] = 10,
    api_key: Optional[str] = None
):
    """
    创建通义千问Rerank函数工厂

    Args:
        model: Rerank模型名称 (默认: gte-rerank-v2)
        top_n: 返回的最高分文档数量 (默认: 10)
        api_key: 通义千问API密钥 (默认从环境变量读取)

    Returns:
        异步rerank函数，签名: async def rerank(query: str, documents: List[str]) -> List[Dict]

    Example:
        >>> rerank_func = create_aliyun_rerank_func()
        >>> results = await rerank_func("查询文本", ["文档1", "文档2"])
        >>> # results = [{"index": 0, "relevance_score": 0.95}, ...]
    """
    async def rerank_func(query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Rerank函数接口（适配LightRAG）

        Args:
            query: 查询文本
            documents: 候选文档列表

        Returns:
            排序后的文档索引和分数列表
        """
        try:
            results = await ali_rerank(
                query=query,
                documents=documents,
                model=model,
                top_n=top_n,
                api_key=api_key
            )
            return results
        except Exception as e:
            from .utils import logger
            logger.warning(f"Rerank调用失败: {e}，返回原始排序")
            # 失败时返回原始顺序和默认分数
            return [{"index": i, "relevance_score": 0.5} for i in range(len(documents))]

    return rerank_func


def create_local_rerank_func(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    top_n: Optional[int] = 10,
    device: str = "cpu",
    model_path: Optional[str] = None
):
    """
    创建本地CrossEncoder Rerank函数工厂

    Args:
        model_name: Rerank模型名称（当model_path为None时使用）
            - "BAAI/bge-reranker-v2-m3" (推荐，支持中英文)
            - "BAAI/bge-reranker-large" (更高精度)
            - "BAAI/bge-reranker-base" (更快速度)
        top_n: 返回的最高分文档数量
        device: 运行设备 ("cpu" 或 "cuda")
        model_path: 本地模型路径（优先使用），如 "D:/models/bge-reranker-v2-m3"

    Returns:
        异步rerank函数，签名: async def rerank(query: str, documents: List[str]) -> List[Dict]

    Example:
        >>> # 使用本地路径
        >>> rerank_func = create_local_rerank_func(model_path="D:/models/bge-reranker-v2-m3")
        >>> # 或使用HuggingFace模型名
        >>> rerank_func = create_local_rerank_func(model_name="BAAI/bge-reranker-v2-m3")
        >>> results = await rerank_func("查询文本", ["文档1", "文档2"])
        >>> # results = [{"index": 0, "relevance_score": 0.95}, ...]
    """
    _model = None
    _actual_model_path = model_path or model_name

    def load_model():
        nonlocal _model
        if _model is None:
            try:
                import os
                # 禁用 mmap 和 fast_gpu 避免 Windows 错误
                os.environ['USE_MMAP'] = '0'
                os.environ['SAFETENSORS_FAST_GPU'] = '0'

                from sentence_transformers import CrossEncoder
                logger.info(f"[Rerank] 加载本地模型: {_actual_model_path}")

                # 尝试加载模型
                try:
                    # 首次尝试：正常加载
                    _model = CrossEncoder(_actual_model_path, device=device, trust_remote_code=True)
                except OSError as oe:
                    if "1455" in str(oe) or "原子" in str(oe) or "atomic" in str(oe).lower():
                        # Windows 大文件内存映射错误，尝试使用 HuggingFace Hub 在线模式
                        logger.warning(f"[Rerank] 本地加载失败（Windows 1455错误），尝试使用 HuggingFace 标识符")
                        # 使用原始模型标识符而不是本地路径
                        _model = CrossEncoder(model_name, device=device, trust_remote_code=True)
                    else:
                        raise

                logger.info("[Rerank] 模型加载完成")
            except ImportError:
                logger.error("sentence_transformers未安装，请运行: pip install -U sentence-transformers")
                _model = None
            except Exception as e:
                logger.error(f"Rerank模型加载失败: {e}")
                _model = None
        return _model

    async def rerank_func(query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Rerank函数接口（适配LightRAG）

        Args:
            query: 查询文本
            documents: 候选文档列表

        Returns:
            排序后的文档索引和分数列表
        """
        model = load_model()
        if model is None:
            # 模型加载失败，返回原始排序
            return [{"index": i, "relevance_score": 0.5} for i in range(len(documents))]

        try:
            # 构建query-doc对
            pairs = [[query, doc] for doc in documents]

            # 计算分数
            scores = model.predict(pairs)

            # 排序并返回结果
            scored = list(enumerate(scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            # 应用top_n限制
            if top_n is not None:
                scored = scored[:top_n]

            return [
                {"index": idx, "relevance_score": float(score)}
                for idx, score in scored
            ]
        except Exception as e:
            logger.warning(f"Rerank计算失败: {e}，返回原始排序")
            return [{"index": i, "relevance_score": 0.5} for i in range(len(documents))]

    return rerank_func


"""Please run this test as a module:
python -m lightrag.rerank
"""
if __name__ == "__main__":
    import asyncio

    async def main():
        # Example usage - documents should be strings, not dictionaries
        docs = [
            "The capital of France is Paris.",
            "Tokyo is the capital of Japan.",
            "London is the capital of England.",
        ]

        query = "What is the capital of France?"

        # Test Jina rerank
        try:
            print("=== Jina Rerank ===")
            result = await jina_rerank(
                query=query,
                documents=docs,
                top_n=2,
            )
            print("Results:")
            for item in result:
                print(f"Index: {item['index']}, Score: {item['relevance_score']:.4f}")
                print(f"Document: {docs[item['index']]}")
        except Exception as e:
            print(f"Jina Error: {e}")

        # Test Cohere rerank
        try:
            print("\n=== Cohere Rerank ===")
            result = await cohere_rerank(
                query=query,
                documents=docs,
                top_n=2,
            )
            print("Results:")
            for item in result:
                print(f"Index: {item['index']}, Score: {item['relevance_score']:.4f}")
                print(f"Document: {docs[item['index']]}")
        except Exception as e:
            print(f"Cohere Error: {e}")

        # Test Aliyun rerank
        try:
            print("\n=== Aliyun Rerank ===")
            result = await ali_rerank(
                query=query,
                documents=docs,
                top_n=2,
            )
            print("Results:")
            for item in result:
                print(f"Index: {item['index']}, Score: {item['relevance_score']:.4f}")
                print(f"Document: {docs[item['index']]}")
        except Exception as e:
            print(f"Aliyun Error: {e}")

    asyncio.run(main())
