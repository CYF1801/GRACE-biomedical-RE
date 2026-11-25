from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# Delimiters
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# Shared validation rules (used by both prompt and code validation)

# Generic terms that should not be extracted as entities
GENERIC_ENTITY_TERMS = [
    "patients", "subjects", "individuals", "cases", "samples",
    "studies", "data", "results", "findings", "analysis",
    "group", "groups", "cohort", "population", "control",
    "experimental", "clinical", "model", "models",
    # Experimental animals/organisms - do not extract as entities
    "rat", "rats", "mouse", "mice", "rabbit", "rabbits",
    "animal", "animals", "organism", "organisms"
]

INVALID_RELATIONSHIP_PATTERNS = [
    # Expression relationships
    " expresses ", "expressed in", "expression pattern",
    # Location relationships
    " found in ", " located in ", " contains ",
    # Developmental/Stage relationships
    "developmental switch", "stage-specific", "neonatal to adult",
    # Experimental subject relationships
    "mouse expresses", "rat has disease", "human samples",
    # Temporal/presence relationships
    " timing coincides ", " occurs at same time ", " present in ", " absent in "
]

PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist extracting entities and relationships from text.

---Task---
Extract entities and relationships to find relationship paths between:
{target_entities_description}

---CRITICAL - TARGET RELATIONSHIP TYPES ONLY---
We are ONLY interested in the following SEVEN relationship types in medical contexts:
1. **treats**: Drug treats disease (e.g., "aspirin treats headache", "imatinib treats leukemia")
2. **causes**: Entity causes another (e.g., "smoking causes lung cancer", "bacteria causes infection")
3. **pharmacological effect**: Drug's effect on body (e.g., "morphine relieves pain", "insulin lowers glucose")
4. **receptor-ligand**: Molecular binding (e.g., "drug binds receptor", "ligand activates receptor")
5. **enzyme-substrate**: Enzymatic relationship (e.g., "enzyme metabolizes substrate", "CYP450 breaks down drug")
6. **drug-disease**: Drug-disease association (positive/negative correlation)
7. **gene-disease**: Gene/protein-disease association (e.g., "PTEN mutation causes tumor", "CCR5 associated with autoimmune diseases")

**DO NOT extract relationships for:**
- Generic "associated with" or "related to" without specific mechanism
- Co-occurrence or correlation without causal link
- "Expresses" or "expression" relationships
- "Found in" or "located in" relationships
- "Both are X" type categorical relationships
- Shared drug target associations

---Instructions---
1. **Entity Extraction Strategy - Path-Focused:**
   - MUST extract: The target entities ({target_entity_list})
   - MUST extract: Intermediate entities that form relationship chains
   - Extract: Entities that directly connect target entities through relationships
   - DO NOT extract: Unrelated entities with no connection to target entities
   - DO NOT extract: Generic terms (e.g., "patients", "subjects", "individuals", "cases", "samples", "studies", "data", "results", "rats", "mice", "animals")

2. **Relationship Extraction - Chain Building:**
   - ONLY extract relationships matching the SEVEN target types above
   - For each relationship, specify the type explicitly in keywords
   - DO NOT extract mere co-occurrence or "associated with" relationships
   - Focus on causal, treatment, and mechanistic relationships

   **VALID RELATIONSHIP KEYWORDS (by type):**
   - treats: treats, therapeutic, therapy, remedy, effective against
   - causes: causes, induces, leads to, results in, triggers
   - pharmacological effect: relieves, reduces, inhibits, activates, improves, prevents
   - receptor-ligand: binds, receptor, ligand, binding, affinity
   - enzyme-substrate: metabolizes, catalyzes, substrate, enzymatic, breakdown
   - drug-disease: correlation, risk, associated (only with clear drug-disease context)
   - gene-disease: associated, implicated, linked, mutation, variant, dysregulation, overexpression, loss (gene-protein affecting disease)

   **CRITICAL - DO NOT CREATE RELATIONSHIPS FOR:**
   - Expression: "A expresses B", "B is expressed in A"
   - Location: "A found in B", "A located in B", "A contains B"
   - Developmental: "developmental switch", "stage-specific"
   - Experimental: "mouse expresses protein", "human samples"
   - Temporal: "A timing coincides with B"
   - Shared target: "both inhibited by X" (unless A→B relationship)

   **Relationship Direction Rules:**
   - Drug → Disease (treats or causes)
   - Gene/Protein → Disease (affects disease)
   - Disease → Symptom (causes symptom)
   - Disease → Disease (complications)
   - Enzyme → Metabolite (catalyzes)
   - Drug → Receptor (binds to)

3. **Path Reconstruction:**
   - Build chains showing how target entities relate through the SEVEN relationship types
   - Include intermediate steps (mediators, mechanisms)
   - Preserve causal and temporal ordering

4. **Format Rules:**
   - Entities: entity{{name}}{{type}}{{description}}
   - Relationships: relation{{source}}{{target}}{{keywords}}{{description}}{{evidence}}{{confidence}}{{strength}}
   - Use {tuple_delimiter} between fields
   - Output entities first, then relationships

5. **Completion:** End with {completion_delimiter}

---Examples---
{examples}
"""

PROMPTS["entity_extraction_user_prompt"] = """Extract entities and relationships from the text below.

<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>
Output format: entity{{name}}{{type}}{{description}} or relation{{...}}
End with: {completion_delimiter}
"""

PROMPTS["entity_continue_extraction_user_prompt"] = """Extract any missed entities and relationships from the previous text.

Do NOT re-output correctly extracted items. Only add missing or corrected items.

End with: {completion_delimiter}
"""

PROMPTS["entity_extraction_examples"] = [
    """<Entity_types>
["Disease","ChemicalOrDrug","GeneOrProtein","Symptom"]

<Input Text>
```
OBJECTIVE: Iodide transport defect (ITD) is a rare disorder characterised by an inability of the thyroid to maintain an iodide gradient across the basolateral membrane of thyroid follicular cells. Treatment with iodide partially corrected the hypothyroidism in patients with this condition.
```

<Output>
entity<|#|>Iodide transport defect<|#|>Disease<|#|>A disorder characterized by the thyroid's inability to maintain iodide gradients, often resulting in congenital hypothyroidism.
entity<|#|>ITD<|#|>Disease<|#|>Abbreviation for iodide transport defect.
entity<|#|>thyroid<|#|>GeneOrProtein<|#|>The thyroid gland affected by the iodide transport defect.
entity<|#|>iodide<|#|>ChemicalOrDrug<|#|>A chemical element used in treatment that partially corrected hypothyroidism.
entity<|#|>hypothyroidism<|#|>Disease<|#|>A condition resulting from iodide transport defect, characterized by reduced thyroid function.
relation<|#|>iodide<|#|>hypothyroidism<|#|>treatment<|#|>Treatment with iodide partially corrected the hypothyroidism<|#|>Treatment with iodide partially corrected the hypothyroidism<|#|>0.9<|#|>STRONG
<|COMPLETE|>

""",
    """<Entity_types>
["Disease","ChemicalOrDrug","GeneOrProtein","Symptom"]

<Input Text>
```
The frequency of polymorphisms involving CYP2E1, GSTP1, and NAT2 were compared between 200 cases of sporadic colorectal cancer and 200 healthy controls. The CYP2E1 c1c2 genotype was associated with increased cancer risk.
```

<Output>
entity<|#|>CYP2E1<|#|>GeneOrProtein<|#|>Cytochrome P450 2E1, an enzyme involved in drug metabolism.
entity<|#|>GSTP1<|#|>GeneOrProtein<|#|>Glutathione S-transferase P1, a detoxification enzyme.
entity<|#|>NAT2<|#|>GeneOrProtein<|#|>N-acetyltransferase 2, a drug-metabolizing enzyme.
entity<|#|>colorectal cancer<|#|>Disease<|#|>A cancer of the colon or rectum.
entity<|#|>CYP2E1 c1c2 genotype<|#|>GeneOrProtein<|#|>A specific genetic variant of CYP2E1.
relation<|#|>CYP2E1<|#|>colorectal cancer<|#|>polymorphism, risk factor<|#|>CYP2E1 polymorphism associated with increased colorectal cancer risk<|#|>CYP2E1 c1c2 genotype was associated with increased cancer risk<|#|>0.85<|#|>STRONG
relation<|#|>CYP2E1 c1c2 genotype<|#|>colorectal cancer<|#|>genotype, risk<|#|>The c1c2 genotype increases risk of colorectal cancer<|#|>c1c2 genotype was associated with increased cancer risk<|#|>0.9<|#|>STRONG
<|COMPLETE|>

""",
    """<Entity_types>
["Disease","ChemicalOrDrug","GeneOrProtein","Symptom"]

<Input Text>
```
BACKGROUND: Methylenetetrahydrofolate reductase (MTHFR) is an important enzyme of folate and methionine metabolism, making it crucial for DNA synthesis and methylation. The objective of this study was to analyze MTHFR gene 677C>T polymorphism in infertile male individuals.
```

<Output>
entity<|#|>MTHFR<|#|>GeneOrProtein<|#|>Methylenetetrahydrofolate reductase, an enzyme involved in folate and methionine metabolism.
entity<|#|>methionine<|#|>ChemicalOrDrug<|#|>An amino acid involved in DNA synthesis and methylation.
entity<|#|>folate<|#|>ChemicalOrDrug<|#|>A B vitamin essential for DNA synthesis and methylation.
entity<|#|>DNA synthesis<|#|>Symptom<|#|>Biological process of creating new DNA molecules.
relation<|#|>MTHFR<|#|>methionine<|#|>enzyme, metabolism, catalytic<|#|>MTHFR is an important enzyme of methionine metabolism, crucial for DNA synthesis and methylation<|#|>MTHFR is an important enzyme of folate and methionine metabolism<|#|>0.95<|#|>STRONG
relation<|#|>MTHFR<|#|>folate<|#|>enzyme, metabolism<|#|>MTHFR is an important enzyme of folate metabolism, crucial for DNA synthesis and methylation<|#|>MTHFR is an important enzyme of folate and methionine metabolism<|#|>0.9<|#|>STRONG
<|COMPLETE|>

""",
    """<Entity_types>
["Disease","ChemicalOrDrug","GeneOrProtein","Symptom"]

<Input Text>
```
BACKGROUND: Imatinib is a tyrosine kinase inhibitor used to treat chronic myeloid leukemia. It inhibits BCR::ABL1, DDR2, and PDGFR kinases. In this study, we profiled imatinib binding affinity across 25 kinases using cellular binding assays. Both ABL1 and DDR2 showed high affinity for imatinib in intact living cells.
```

<Output>
entity<|#|>imatinib<|#|>ChemicalOrDrug<|#|>A tyrosine kinase inhibitor used to treat chronic myeloid leukemia.
entity<|#|>BCR::ABL1<|#|>GeneOrProtein<|#|>Fusion oncogene tyrosine kinase, target of imatinib in CML.
entity<|#|>DDR2<|#|>GeneOrProtein<|#|>Discoidin domain receptor 2 tyrosine kinase.
entity<|#|>PDGFR<|#|>GeneOrProtein<|#|>Platelet-derived growth factor receptor.
entity<|#|>chronic myeloid leukemia<|#|>Disease<|#|>A cancer of the white blood cells.
relation<|#|>imatinib<|#|>BCR::ABL1<|#|>inhibits, treats<|#|>Imatinib inhibits BCR::ABL1 and is used to treat CML<|#|>Imatinib is a tyrosine kinase inhibitor used to treat chronic myeloid leukemia. It inhibits BCR::ABL1<|#|>0.95<|#|>STRONG
relation<|#|>imatinib<|#|>DDR2<|#|>inhibits, binds<|#|>Imatinib showed high affinity binding to DDR2<|#|>Both ABL1 and DDR2 showed high affinity for imatinib<|#|>0.85<|#|>MODERATE
relation<|#|>BCR::ABL1<|#|>chronic myeloid leukemia<|#|>causes, drives<|#|>BCR::ABL1 drives chronic myeloid leukemia<|#|>Imatinib is used to treat chronic myeloid leukemia. It inhibits BCR::ABL1<|#|>0.9<|#|>STRONG
<|COMPLETE|>
NOTE: DO NOT extract ABL1 ↔ DDR2 relationship just because both are inhibited by imatinib. This is a shared drug target association, NOT a direct biological relationship.
""",
]

PROMPTS["summarize_entity_descriptions"] = """Synthesize these descriptions into one coherent summary.

Entity: {description_name}

Descriptions:
{description_list}

Summary (max {summary_length} tokens, {language}):
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question.[no-context]"

PROMPTS["rag_response"] = """Answer the query using ONLY the provided context.

Query: {query}

Context:
{context_data}

Instructions:
- Use ONLY information from context
- Cite relevant documents with [n] references
- Output in {response_type}
- Same language as query

Answer:"""

PROMPTS["naive_rag_response"] = """Answer the query using ONLY the provided document chunks.

Query: {query}

Context:
{content_data}

Instructions:
- Use ONLY information from context
- Cite relevant documents with [n] references
- Output in {response_type}
- Same language as query

Answer:"""

PROMPTS["kg_query_context"] = """
Knowledge Graph Entities:
{entities_str}

Knowledge Graph Relationships:
{relations_str}

Document Chunks:
{text_chunks_str}

References:
{reference_list_str}
"""

PROMPTS["naive_query_context"] = """
Document Chunks:
{text_chunks_str}

References:
{reference_list_str}
"""

PROMPTS["keywords_extraction"] = """Extract keywords from the query.

Query: {query}

Output JSON (high_level and low_level keywords):
{{
  "high_level_keywords": ["concept1", "concept2"],
  "low_level_keywords": ["entity1", "entity2"]
}}

Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Query: "How does international trade influence global economic stability?"
{{
  "high_level_keywords": ["International trade", "Global economic stability"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange"]
}}
""",
]

# Prompt Manager for loading YAML-based prompts
import os
import yaml

def get_prompt_manager():
    """Get prompt manager that loads prompts from YAML files"""
    class PromptManager:
        def __init__(self):
            self.prompts = {}
            # Hardcoded CoT prompt to avoid YAML parsing issues
            self.prompts = {
                'syract_enhanced_with_kg': """==MOST CRITICAL: Your "reason" field MUST follow this EXACT format==
reason = "Step 0: [X]. Step 1: [Y]. Step 2: [Z]. Step 3: [W]. Step 4: [calc]. Step 5: [fusion]. Step 6: [final]. Final: [answer with conf]"

==EXAMPLES (COPY THESE FORMATS)==
Immediate rejection:
"Step 0: Entity 'rat' is a model organism -> immediate rejection (confidence 0.2). Steps 1-6 skipped. Final: answer=no, confidence=0.2"

Full analysis:
"Step 0: Passed. Step 1: Document shows 'inhibits' -> score 0.8. Step 2: Passed. Step 3: Graph shows direct path -> baseline 0.8. Step 4: text_conf = 0.8+0.1 = 0.9. Step 5: |0.9-0.9|<=0.3 -> fusion = 0.6*0.9+0.4*0.9 = 0.9. Step 6: 0.9>0.5 -> answer=yes. Final: YES with confidence 0.9"

==ENTITY INFO==
Entity A: {entity_a} (Type: {entity_a_type})
Entity B: {entity_b} (Type: {entity_b_type})

==INPUT DATA==
Knowledge Graph: {knowledge_graph}
Query Document: {query_document}
Graph Path Confidence: {graph_score}

==STEPS TO EXECUTE (include analysis in reason)==

Step 0: Check for generic entities (patient, sample, human, subjects) or model organisms (mouse, rat, mice)
- If found -> "Step 0: Entity '[X]' is [type] -> rejection (confidence 0.2). Steps 1-6 skipped. Final: NO, confidence=0.2"
- If not found -> "Step 0: Passed"

Step 1: Check document for direct mechanism (binds, inhibits, activates, causes, treats)
- If explicit mechanism -> score 0.7-0.9
- If "associated with" only -> score <= 0.5
- Output -> "Step 1: [Found/Not found] -> score [X]"

Step 2: Check exclusions (hierarchy, diagnostic, indirect, shared pathway)
- If exclusion -> "Step 2: Exclusion - [X] -> cap at 0.3"
- If not -> "Step 2: Passed"

Step 3: Check Knowledge Graph for known relationships
- Output -> "Step 3: Graph shows [relationship type] -> baseline [X]"

Step 4: Calculate text_conf = Step3 + Step1 adjustment
- Output -> "Step 4: text_conf = [X] + [Y] = [Z]"

Step 5: Graph-LLM fusion
- Calculate: diff = |text_conf - graph_score|
- If diff <= 0.3: final = 0.6 * text_conf + 0.4 * graph_score
- If diff > 0.3: explain decision
- Output -> "Step 5: |[Z]-[graph_score]| = [diff] -> fusion = [calc] = [final]"

Step 6: Final decision
- If final > 0.5 -> answer="yes"
- If final <= 0.5 -> answer="no"
- Output -> "Step 6: [final] [comparison] -> answer=[yes/no]"

==REMEMBER==
Your reason MUST include ALL executed steps in the format shown in EXAMPLES above.
""",
                'basic': "Verify if {entity_a} and {entity_b} have a meaningful medical relationship."
            }
            print("[PromptManager] Using hardcoded CoT prompt")

        def get_prompt(self, category='relation_verification', variant='basic'):
            """Get prompt by category and variant"""
            if category == 'relation_verification':
                return self.prompts.get(variant, self.prompts.get('basic', "Verify relationship between {entity_a} and {entity_b}"))
            else:
                return PROMPTS.get(f"{category}_prompt", "Default prompt")

    return PromptManager()
