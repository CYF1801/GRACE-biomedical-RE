# GRACE: Graph-and-Retrieval Augmented Confidence Estimation for Biomedical Relation Verification

基于检索增强 + 临时知识图谱 + LLM Chain-of-Thought 的**生物医学实体关系验证管线**（二分类：两个实体之间是否存在有意义的生物学/医学关系）。

在 BioRED 基准 896 对全量测试上，RRF 混合检索（hybrid，BM25 + 向量召回融合）取得 **F1 = 0.7034**（P = 0.5822，R = 0.8881），较单路向量检索（baseline，F1 = 0.6762）高 **+2.72 pp**，精确率与召回率同步占优；多模型验证消融（GLM-5.2，验证层）F1 = 0.7478，精确率 0.6532。全部成绩由本仓库代码产出，逐行结果与 MD5 见 [`results_provenance/RESULTS_PROVENANCE.md`](results_provenance/RESULTS_PROVENANCE.md)。

## 架构

```
test_biored_rrf_hybrid.py      实验入口：抽样 / RRF monkey-patch / 逐对调度 / 结果CSV / 报告
        │
lightrag_core.py               验证引擎：摘要检索 → 临时知识图谱构建 → 多跳路径搜索
        │                      → 图谱置信分（调和平均×跳数惩罚）
        │                      → 7步 CoT prompt → LLM 裁决 (YES/NO + confidence)
        ▼
syRACT_dynamic_rag_with_omop.py  检索与缓存层：PubMed E-utilities、PMID/ embedding 双层缓存、
                                 MeSH 规范化、gte-rerank-v2 精排
```

**判定规则**：图谱分（代码计算）与文本分（LLM 按动词强度规则打分）按一致性情形融合
（无图谱→纯文本；|差|≤0.3→0.6/0.4 加权；冲突→按谓词类型采信），confidence > 0.5 判 YES。

## 仓库结构

```
├── syRACT_dynamic_rag_with_omop.py   # 检索与缓存层
├── lightrag_core.py                  # 关系验证引擎
├── test_biored_rrf_hybrid.py         # BioRED 实验脚本（baseline / hybrid / report 模式）
├── test_cdr.py / test_cdr_simple.py  # CDR 实验（双臂对照：CDR域约束 vs 原生prompt；_simple 为无图谱版）
├── test_gda.py / test_gda_simple.py  # GDA 实验（GDA域约束注入，同上）
├── biored_test_samples.csv           # BioRED test 集（946 行 = 896 个唯一实体对，公开基准数据）
├── data/CDR/test_new.json            # CDR 基准（BioCreative V CDR 公开数据）
├── data/GDA/test.json                # GDA 基准（公开数据）
├── lightrag_patch/lightrag/          # 对官方 LightRAG 1.4.16 的定制补丁（覆盖到安装包同路径）
│   ├── prompt.py                     # 抽取模板：关系记录扩展为 8 字段（evidence/confidence/strength 等）+ 共享过滤规则
│   ├── operate.py                    # 关系抽取：置信度解析、<0.3 边过滤、多证据按 max 合并入图
│   ├── rerank.py                     # 新增 create_aliyun_rerank_func（gte-rerank-v2 / DashScope）
│   ├── llm/openai.py                 # DashScope workspace 请求头支持
│   └── prompts/__init__.py           # 定制关系验证 prompt（syract_enhanced_with_kg 变体）
├── requirements.txt                  # 实验环境精确版本
├── install_patch.py                  # LightRAG 补丁一键覆盖脚本
├── results_provenance/               # 论文成绩溯源：896 全量结果文件、样本清单、MD5 与版本说明
└── .env.example                      # 环境变量模板
```

**域约束 prompt 机制**：CDR/GDA 实验不改 prompt 模板本身——脚本内的
`cdr_llm_func` / `gda_llm_func` 包装器仅拦截最终关系验证调用，在用户
prompt 前部注入实体类型约束（CDR：A 恒为化学物/药物、B 恒为疾病；GDA：
A 恒为基因、B 恒为疾病），图谱抽取等其余调用原样透传，臂间唯一差异即
验证 prompt，保证严格配对。

## 数据

| 文件 | 说明 |
|---|---|
| `biored_test_samples.csv` | BioRED test 集实体对全集：946 行 = 896 个唯一实体对（同一对可含多种关系类型）。来源：BioRED 公开基准（Xing et al., 2024）test 集，摘要经 PubMed E-utilities 获取。列为 pmid / entity1 / entity2 / entity1_text / entity2_text / entity1_type / entity2_type / abstract / true_label / relation_type。MD5：9478c3cd6bd1b244cba06a0a493ef08f |
| `data/CDR/test_new.json` | CDR 基准（BioCreative V CDR 公开数据） |
| `data/GDA/test.json` | GDA 基准（公开数据） |
| `results_provenance/manifest_n896_seed42.json` | 论文计分使用的固定样本清单（896 对，seed=42），与 `RESULTS_PROVENANCE.md` 配套 |

脚本运行时按 (pmid, entity1_text, entity2_text, relation_type) 四字段去重得到 896 个唯一样本，并以 seed=42 确定性抽样；固定清单自动生成于 `results/`（`manifest_n896_seed42.json`），与发布清单逐键一致。全部数据为公开基准，新用户无需作者本地缓存即可从头运行。

## 环境与安装

**测试环境**：Python 3.13 + Windows 11（开发与实际运行平台；依赖均为跨平台包，其他平台未系统验证）。

```bash
# 1) 安装依赖（含 lightrag-hku==1.4.16）
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"

# 2) 一键覆盖 LightRAG 定制补丁（prompt.py / operate.py / rerank.py / llm/openai.py / prompts/__init__.py）
python install_patch.py

# 3) 配置密钥
cp .env.example .env   # 填入 DASHSCOPE_API_KEY 与 NCBI_EMAIL
```

依赖基于 [LightRAG](https://github.com/HKUDS/LightRAG)（HKUDS）二次开发：
先从 PyPI 安装 `lightrag-hku==1.4.16`，再用 `install_patch.py` 覆盖补丁。
其中 `prompts/__init__.py` 包含本项目定制的 `syract_enhanced_with_kg`
关系验证 prompt 变体；`prompt.py` / `operate.py` 为图谱构建所需的置信度
字段扩展（官方 1.4.16 不含，缺失则图谱打分退化、无法复现论文成绩）；
`rerank.py` 提供官方版没有的 `create_aliyun_rerank_func`（主程序直接
import，缺失将无法启动）。

**模型与外部服务**：

| 组件 | 默认模型/服务 | 用途 | 访问条件 |
|---|---|---|---|
| LLM | `qwen-plus`（`.env` 的 `QWEN_MODEL` 可换） | 图谱三元组抽取 + 关系验证 | `DASHSCOPE_API_KEY`（阿里云 DashScope，国内直连） |
| Embedding | `text-embedding-v3`（1024 维） | 摘要与查询向量化 | 同上 |
| Rerank | `gte-rerank-v2` | 检索精排 | 同上 |
| PubMed | NCBI E-utilities 公共接口 | 摘要获取 | `NCBI_EMAIL` 必填；`NCBI_API_KEY` 可选（提升限流配额） |

所有密钥经 `.env` 注入（模板见 `.env.example`），仓库不含真实密钥。
`QWEN_MODEL` / `QWEN_EMBEDDING_MODEL` 及各路径（`GRACE_CACHE_DIR` /
`GRACE_TEMP_DIR` / `GRACE_RESULTS_DIR` / `BIORED_CSV`）均有默认值，
用途与默认值详见 `.env.example` 内注释。

## 运行

> 注意：阶段参数解析保留**最后一个**阶段参数——baseline 与 hybrid 请分两条命令执行，不要写在同一条命令里。

```bash
# ① 验证安装（3 对冒烟：确认环境、密钥与 PubMed/DashScope 连通性）
python test_biored_rrf_hybrid.py 3 hybrid

# ② baseline 臂（896 对，论文计分口径）
python test_biored_rrf_hybrid.py 896 baseline

# ③ hybrid 臂（896 对；与 ② 使用同一次实验目录 results/，两臂结果按臂分文件）
python test_biored_rrf_hybrid.py 896 hybrid

# ④ 统计报告（两臂 P/R/F1、混淆矩阵、McNemar 精确检验、逐样本对照）
python test_biored_rrf_hybrid.py report
```

- **同一次实验目录**：②③ 默认都写入 `results/`（`baseline_results.csv` /
  `hybrid_results.csv` / `comparison_report.txt`；目录可用 `GRACE_RESULTS_DIR`
  覆盖）。两臂先后顺序不限，但请各自独立成命令执行。
- 中断后重新执行同一命令，即从剩余样本继续。

**冒烟预期输出**：每个样本依次打印实体、金标准与
`判断: YES/NO | 真实: YES/NO | ✓/✗ | confidence=… | graph=… | 耗时`，
结束后在 `results/` 下生成 `hybrid_results.csv`（列为 mode / idx / pmid /
entity_a / entity_b / true_label / predicted / correct / confidence /
llm_score / graph_score / multi_hop_path / relation_type / elapsed_sec /
graph_pmids / graph_source_mode）与确定性抽样清单 `manifest_n3_seed42.json`。
3 对冒烟全部出判定即说明环境与输入格式正常；**论文计分请使用 896 全量口径**。

## 从安装到复现论文成绩的完整流程

1. 按"环境与安装"完成依赖、补丁与 `.env` 配置
2. 冒烟：`python test_biored_rrf_hybrid.py 3 hybrid`——3 对全部出判定即环境正常
3. 双臂全量：分别执行②③（896 对各一遍，约 2-4 分钟/对，两臂可分日运行）
4. 统计：④——得到两臂 P/R/F1、混淆矩阵、McNemar 精确检验与逐样本对照报告
5. 与论文成绩核对：`results_provenance/RESULTS_PROVENANCE.md` 附有 10 行指标
   重算脚本，可直接对随仓库发布的 896 行逐样本结果文件重算 P/R/F1
   （hybrid 预期 0.7034，baseline 预期 0.6762）

## 数据与隐私

- 实验数据为公开基准 BioRED（Xing et al., 2024）的 test 集实体对；
  PubMed 检索仅使用公开 E-utilities 接口。
- 所有密钥经环境变量注入，仓库不含任何凭据；`.env` 已在 `.gitignore` 中排除。

## License

本项目按 **Apache License 2.0** 发布（见 [`LICENSE`](LICENSE)）；代码基于
[LightRAG](https://github.com/HKUDS/LightRAG)（上游同为 Apache-2.0）二次开发。
生物医学用途请自行评估。
#   G R A C E - b i o m e d i c a l - R E  
 