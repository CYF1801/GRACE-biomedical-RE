"""
LightRAG Prompts Module

Provides shared prompt management for relation validation
across evaluation and production systems.
"""

from typing import Any, Optional
from threading import Lock


class PromptManager:
    """
    Singleton manager for serving the built-in prompt set.

    Ensures single-source prompt management across all systems.
    """

    _instance: Optional['PromptManager'] = None
    _lock = Lock()
    _prompts: dict[str, Any] = {}

    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the prompt manager with the built-in prompt set."""
        if not self._prompts:
            self._prompts = self._get_default_prompts()

    def get_prompt(self, prompt_name: str, variant: str = 'basic') -> str:
        """
        Get a prompt by name and variant.

        Args:
            prompt_name: Name of the prompt (e.g., 'relation_verification')
            variant: Variant key ('basic' or 'detailed')

        Returns:
            The prompt template string
        """
        if prompt_name in self._prompts:
            variants = self._prompts[prompt_name]
            if isinstance(variants, dict):
                return variants.get(variant, variants.get('basic', ''))
            return str(variants)
        return ""

    def get_all_prompts(self) -> dict[str, Any]:
        """Get all loaded prompts."""
        return self._prompts.copy()

    def get_variants(self, prompt_name: str) -> list[str]:
        """Get available variants for a prompt."""
        if prompt_name in self._prompts:
            variants = self._prompts[prompt_name]
            if isinstance(variants, dict):
                return list(variants.keys())
        return []

    @staticmethod
    def _get_default_prompts() -> dict[str, Any]:
        """Get the built-in prompt set with CoT format."""
        return {
            "relation_verification": {
                "syract_enhanced_with_kg": """==MOST CRITICAL: Your "reason" field MUST follow this EXACT format==
reason = "Step 0: [X]. Step 1: [Y]. Step 2: [Z]. Step 3: [W]. Step 4: [calc]. Step 5: [fusion]. Step 6: [final]. Final: [answer with conf]"

==EXAMPLES (COPY THESE FORMATS)==
Immediate rejection:
"Step 0: Entity 'rat' is a model organism -> immediate rejection (confidence 0.2). Steps 1-6 skipped. Final: answer=no, confidence=0.2"

Strong mechanism (YES):
"Step 0: Passed. Step 1: Document shows 'drug inhibits protein' -> score 0.8. Step 2: Passed. Step 3: Graph shows direct path with predicate 'inhibits', Graph Path Confidence = 0.90. Step 4: text_conf = 0.8 (pure text). Step 5: |0.8-0.9| = 0.1 <= 0.3 -> fusion = 0.6*0.8+0.4*0.9 = 0.84. Step 6: 0.84>0.5 -> answer=yes. Final: YES with confidence 0.84"

Weak association only (NO):
"Step 0: Passed. Step 1: Document shows 'X increased Y' and 'Y associated with Z' -> only correlation words -> score 0.3. Step 2: Correlation only, not direct causation between X and Z -> cap at 0.4. Step 3: No usable graph evidence (Graph Path Confidence = 0.00). Step 4: text_conf = 0.3 (pure text). Step 5: No graph evidence -> skip fusion, final = text_conf = 0.3. Step 6: 0.3<=0.5 -> answer=no. Final: NO with confidence 0.3"

Conditional change (NO):
"Step 0: Passed. Step 1: Document shows 'Y expression reduced under X treatment' -> conditional change, not direct mechanism -> score 0.25. Step 1.5: Passed (entities match). Step 2: Correlation observed only under specific condition, not direct mechanism -> cap at 0.3. Step 3: No usable graph evidence (Graph Path Confidence = 0.00). Step 4: text_conf = 0.25 (pure text). Step 5: No graph evidence -> skip fusion, final = text_conf = 0.25. Step 6: 0.25<=0.5 -> answer=no. Final: NO with confidence 0.25"

Graph-text conflict, weak graph predicate (NO):
"Step 0: Passed. Step 1: Document shows 'X associated with Y' -> only association words -> score 0.3. Step 2: Correlation only, not direct causation -> cap at 0.4. Step 3: Graph shows direct path with predicate 'associated', Graph Path Confidence = 0.90. Step 4: text_conf = 0.3 (pure text). Step 5: |0.3-0.9| = 0.6 > 0.3 -> conflict. Graph predicate is generic association, not a clear mechanism -> trust text: final = 0.7*0.3+0.3*0.9 = 0.48. Step 6: 0.48<=0.5 -> answer=no. Final: NO with confidence 0.48"

Graph-text conflict, clear graph mechanism (YES):
"Step 0: Passed. Step 1: Query Document mentions the pair without explicit verbs -> score 0.2. Step 2: Passed. Step 3: Graph shows direct path with predicate 'inhibits' (clear mechanism verb, from other retrieved literature), Graph Path Confidence = 0.90. Step 4: text_conf = 0.2 (pure text). Step 5: |0.2-0.9| = 0.7 > 0.3 -> conflict. Graph predicate 'inhibits' is a clear mechanism consistent with the Step 1 verb list -> trust graph: final = 0.4*0.2+0.6*0.9 = 0.62. Step 6: 0.62>0.5 -> answer=yes. Final: YES with confidence 0.62"

Entity mismatch (NO):
"Step 0: Passed. Step 1: Document shows 'drug treats Tic disorders' but query asks about psychosis -> score 0.5. Step 1.5: Entity mismatch - document mentions Tic disorders, not psychosis -> cap at 0.4. Step 2: Passed. Step 3: Graph shows direct path, Graph Path Confidence = 0.60. Step 4: text_conf = 0.4 (pure text). Step 5: |0.4-0.6| = 0.2 <= 0.3 -> fusion = 0.6*0.4+0.4*0.6 = 0.48. Step 6: 0.48<=0.5 -> answer=no. Final: NO with confidence 0.48"

Hierarchy relation (NO):
"Step 0: Entity A 'congenital hypothyroidism' OK; Entity B 'hypothyroidism' OK -> Passed. Step 1: Document describes 'congenital hypothyroidism' as a subtype of 'hypothyroidism' -> score 0.3. Step 2: Hierarchy - one entity is a subtype/stage of the other, not an independent relation -> cap at 0.3. Step 3: Graph shows direct path, Graph Path Confidence = 0.90. Step 4: text_conf = 0.3 (pure text). Step 5: |0.3-0.90| = 0.6 > 0.3 -> conflict. The graph path encodes the hierarchy itself, not an independent mechanism -> trust text: final = 0.7*0.3+0.3*0.90 = 0.48. Step 6: 0.48<=0.5 -> answer=no. Final: NO with confidence 0.48"

==ENTITY INFO==
Entity A: {entity_a} (Type: {entity_a_type})
Entity B: {entity_b} (Type: {entity_b_type})

==INPUT DATA==
Knowledge Graph: {knowledge_graph}
Query Document: {query_document}
Graph Path Confidence: {graph_score}

==STEPS TO EXECUTE (include analysis in reason)==

Step 0: MANDATORY entity screening - you MUST output a separate verdict for EACH entity (A and B), never skip
- Generic research terms: patient, sample, human, subjects, participants, volunteers, cells, tissue, serum
- Model organisms: mouse, rat, mice, drosophila, c. elegans, zebrafish
- Cell lines: COS-7, COS7, HeLa, HEK293, CHO, 3T3, and any other named immortalized cell line
- If EITHER entity falls in the above categories -> "Step 0: Entity '[X]' is [generic term|model organism|cell line] -> rejection (confidence 0.2). Steps 1-6 skipped. Final: NO, confidence=0.2"
- If BOTH pass -> "Step 0: Entity A '[X]' OK; Entity B '[Y]' OK -> Passed"

Step 1: Check document for relationship strength
- STRONG mechanism verbs (binds, inhibits, activates, treats, blocks, induces, prevents) -> score 0.7-0.9
- WEAK association words (increased, decreased, associated with, links to, related, correlated, elevated) -> score 0.2-0.4
- CONDITIONAL changes (X under/treated/with Y shows Z, Y expression under X treatment) -> score 0.0-0.3 (correlation, not mechanism)
- NO direct relationship -> score 0.0
- Output -> "Step 1: [verb type] -> score [X]"

Step 1.5: Check entity matching
- If document mentions a different but related entity instead of the queried entity -> "Step 1.5: Entity mismatch - document mentions [X] not [queried entity] -> cap at 0.4"
- If exact match -> "Step 1.5: Passed"

Step 2: Check for correlation vs causation and relationship types
- Correlation/association (not direct causation between A and B) -> "Step 2: Correlation only, not causation -> cap at 0.4"
- Symptom/complication (Y is symptom of X, or X causes symptom Y) -> "Step 2: Y is symptom/complication of X, not independent relationship -> cap at 0.3"
- Side effect/adverse reaction (drug X causes side effect Y) -> "Step 2: Side effect/adverse reaction, not direct therapeutic relationship -> cap at 0.3"
- Experimental context (observed only in specific conditions) -> "Step 2: Observed only in specific experimental context -> cap at 0.3"
- Treatment indication (drug X is used to treat Y) -> "Step 2: Passed (therapeutic mechanism)"
- Exclusion (hierarchy, diagnostic, indirect, shared pathway) -> "Step 2: Exclusion - [X] -> cap at 0.3"
  * hierarchy = one entity is a subtype/stage/form/strain of the other (e.g., 'congenital hypothyroidism' vs 'hypothyroidism'; 'nodular goitre' vs 'goitre'; 'HPV16' vs 'HPV'); such pairs are NOT independent relations
- Direct mechanism -> "Step 2: Passed"

Step 3: Describe the Knowledge Graph path (DESCRIPTIVE ONLY - this step does NOT enter any calculation)
- If graph evidence exists -> "Step 3: Graph shows [path and predicate], Graph Path Confidence = [X]"
- If Knowledge Graph is empty / only states no path was found / Graph Path Confidence = 0.00 -> "Step 3: No usable graph evidence (Graph Path Confidence = 0.00)"

Step 4: Calculate text_conf from TEXT evidence ONLY (do NOT add any graph value here)
- text_conf = Step 1 score, after applying any Step 1.5 / Step 2 caps
- Output -> "Step 4: text_conf = [value] (pure text)"

Step 5: Graph-LLM fusion (choose exactly ONE case)
- Case A - No graph evidence (Graph Path Confidence = 0.00):
  -> Skip fusion. final = text_conf
  -> "Step 5: No graph evidence -> pure text judgment, final = text_conf = [Z]"
- Case B - diff = |text_conf - graph_score| <= 0.3:
  -> final = 0.6 * text_conf + 0.4 * graph_score
  -> "Step 5: |[Z]-[graph_score]| = [diff] <= 0.3 -> fusion = 0.6*[Z]+0.4*[graph_score] = [final]"
- Case C - diff = |text_conf - graph_score| > 0.3 (conflict):
  -> If every hop of the graph path has a clear mechanism predicate (binds, inhibits, activates, blocks, induces, prevents, treats, causes, metabolizes) -> trust graph: final = 0.4 * text_conf + 0.6 * graph_score
  -> Otherwise -> trust text: final = 0.7 * text_conf + 0.3 * graph_score
  -> State in one sentence which side you trusted and why

Step 6: Final decision
- If final > 0.5 -> answer="yes"
- If final <= 0.5 -> answer="no"
- Output -> "Step 6: [final] [comparison] -> answer=[yes/no]"

==REMEMBER==
Your reason MUST include ALL executed steps in the format shown in EXAMPLES above.
- text_conf must come from Step 1 text analysis ONLY; the graph score is fused ONLY in Step 5.
- When Graph Path Confidence = 0.00, you MUST use Case A (pure text judgment); never multiply text confidence by 0.6.
- Step 0 verdicts are required for BOTH entities individually; Step 2 requires an explicit hierarchy/diagnostic verdict before any "Passed"; never skip these checks even when the graph shows a high-confidence path.
""",
                "basic": "Verify if {entity_a} and {entity_b} have a meaningful medical relationship."
            }
        }


# Global instance
_global_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """
    Get the global PromptManager instance.

    Returns:
        The global PromptManager singleton instance
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = PromptManager()
    return _global_manager


__all__ = [
    'PromptManager',
    'get_prompt_manager'
]
