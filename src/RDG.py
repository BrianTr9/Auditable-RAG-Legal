"""
Retrieval-Dependent Grammars (RDG) - Layer 2
============================================

Grammar-constrained citation generation for auditable RAG.

Core Features:
- GBNF Grammar: Token-level constraints from retrieved documents
- Structured Chain-of-Thought: Enforces reasoning → answer flow
- Zero-Syntatic-Citation-Hallucination: Set(Retrieved) ∩ Set(Reasoning) ∩ Set(Generated)
- Auto-Citation Collection: Final citations derived from reasoning steps

Author: Trung Bao Truong
Date: February 2026
"""

import os
import re
import json
import sys
import io
from typing import List, Dict, Optional, Any, Literal, Tuple
from dataclasses import dataclass, field
from langchain_core.documents import Document

# Public API exports
__all__ = [
    'Citation',
    'ReasoningStep',
    'StructuredReasoning',
    'StructuredOutput',
    'GBNFGrammarBuilder',
    'ReferenceExtractor',
    'collect_validated_citations',
    'RDGPipeline',
    'get_rdg_pipeline',
    'get_rdg_mode',
]

# ==========================================
# SETUP & CONFIG
# ==========================================
try:
    from llama_cpp import Llama, LlamaGrammar
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None
    LlamaGrammar = None

DEFAULT_MODEL_PATH = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
DEFAULT_N_CTX = 16384

# ==========================================
# DATA MODELS
# ==========================================

@dataclass
class Citation:
    reference_id: str   # Full ID: doc_0_Section_I_TITLE
    section_num: str    # I, 2.1
    section_title: str  # TITLE
    contract_id: str    # doc_0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "section_num": self.section_num,
            "section_title": self.section_title
        }

    @staticmethod
    def from_reference_string(ref: str) -> 'Citation':
        section_marker = "_Section_"
        if section_marker not in ref:
            return Citation(ref, "Unknown", "Unknown", "Unknown")
        try:
            contract_and_prefix, rest = ref.split(section_marker, 1)
            if '_' in rest:
                section_num, section_title = rest.split('_', 1)
            else:
                section_num = rest
                section_title = "Unknown"
            return Citation(ref, section_num, section_title, contract_and_prefix)
        except:
            return Citation(ref, "Unknown", "Unknown", "Unknown")
    
    def to_short_id(self) -> str:
        """Compact ID for reasoning steps: doc_0_sec_I"""
        if self.contract_id.startswith("doc_"):
            parts = self.contract_id.split("_", 2)
            doc_prefix = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else self.contract_id
        else:
            doc_prefix = self.contract_id
        return f"{doc_prefix}_sec_{self.section_num}"
    
    @staticmethod
    def get_short_id_from_reference(ref: str) -> str:
        try:
            return Citation.from_reference_string(ref).to_short_id()
        except:
            return ref

@dataclass
class ReasoningStep:
    extracted_snippet: str
    deduction: str 
    step_type: str = "inference"
    citations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "extracted_snippet": self.extracted_snippet,
            "deduction": self.deduction,
            "type": self.step_type,
            "citations": self.citations
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'ReasoningStep':
        if not isinstance(data, dict): 
            # Fallback for legacy string format or non-dict input
            return ReasoningStep(extracted_snippet="", deduction=str(data))
        
        # Handle backward compatibility if "statement" exists but new fields don't
        snippet = data.get("extracted_snippet", "")
        deduction = data.get("deduction", "")
        if "statement" in data and not snippet and not deduction:
            deduction = data["statement"]

        return ReasoningStep(
            extracted_snippet=snippet,
            deduction=deduction,
            step_type=data.get("type", "inference"),
            citations=data.get("citations", [])
        )

@dataclass
class StructuredReasoning:
    steps: List[ReasoningStep]
    
    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}
    
    @staticmethod
    def from_dict(data: Dict) -> 'StructuredReasoning':
        if not isinstance(data, dict): return StructuredReasoning([])
        return StructuredReasoning([ReasoningStep.from_dict(s) for s in data.get("steps", [])])

@dataclass
class StructuredOutput:
    citations: List[Citation]
    reasoning: StructuredReasoning 
    answer: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "reasoning": self.reasoning.to_dict(),
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations]
        }
        if self.metadata: result["metadata"] = self.metadata
        return result

    @staticmethod
    def from_dict(data: Dict) -> 'StructuredOutput':
        citations = []
        for c in data.get("citations", []):
            try:
                if isinstance(c, dict):
                    if "reference_id" in c:
                        citations.append(Citation.from_reference_string(c["reference_id"]))
                    elif all(k in c for k in ["contract_id", "section_num", "section_title"]):
                        ref = f"{c['contract_id']}_Section_{c['section_num']}_{c['section_title']}"
                        citations.append(Citation.from_reference_string(ref))
                elif isinstance(c, str):
                    citations.append(Citation.from_reference_string(c))
            except: pass
        
        return StructuredOutput(
            citations=citations,
            reasoning=_parse_reasoning(data.get("reasoning", "")),
            answer=data.get("answer", "")
        )

# ==========================================
# GRAMMAR & HELPERS
# ==========================================

def _gbnf_quoted_id(source_id: str) -> str:
    """GBNF rule for one source ID emitted as a JSON string: '"\\"" "<escaped id>" "\\""'.

    Escaping is done outside the f-string: a backslash inside an f-string expression is a
    SyntaxError before Python 3.12, and the HPC environment runs Python 3.10.
    """
    escaped = source_id.replace("\\", "\\\\").replace('"', '\\"')
    return f'"\\"" "{escaped}" "\\""'


class GBNFGrammarBuilder:
    def build_structured_cot_grammar(self, valid_citations: List[str], answer_mode: Literal["freeform", "yn"] = "freeform") -> str:
        """Constructs GBNF grammar for Structured CoT.

        answer_mode:
            - "freeform": answer is an arbitrary JSON string
            - "yn": answer is strictly "Y" or "N" (as a JSON string)
        """
        if not valid_citations:
            full_enum = short_enum = '"\"" "no_citation" "\""'
        else:
            full_enum = ' | '.join([_gbnf_quoted_id(c) for c in valid_citations])
            # Short IDs for steps
            short_ids = [Citation.get_short_id_from_reference(c) for c in valid_citations]
            short_enum = ' | '.join([_gbnf_quoted_id(c) for c in short_ids])

        answer_rule = "string" if answer_mode == "freeform" else "yn-string"
        
        # Streamlined Grammar: Snippet -> Deduction -> Type -> Citations
        # Enforces Mechanical Grounding before Logical Deduction
        return f'''
root ::= "{{" ws reasoning-field ws "," ws answer-field ws "}}"

reasoning-field ::= "\\"reasoning\\"" ws ":" ws reasoning-object
reasoning-object ::= "{{" ws "\\"steps\\"" ws ":" ws steps-array ws "}}"
steps-array ::= "[]" | "[" ws reasoning-step (ws "," ws reasoning-step)* ws "]"
reasoning-step ::= "{{" ws "\\"extracted_snippet\\"" ws ":" ws string ws "," ws "\\"deduction\\"" ws ":" ws string ws "," ws "\\"type\\"" ws ":" ws step-type ws "," ws "\\"citations\\"" ws ":" ws short-citation-list ws "}}"

short-citation-list ::= "[]" | "[" ws short-citation (ws "," ws short-citation)* ws "]"
short-citation ::= {short_enum}
step-type ::= "\\"premise\\"" | "\\"inference\\"" | "\\"conclusion\\""

answer-field ::= "\\"answer\\"" ws ":" ws {answer_rule}

yn-string ::= "\\"Y\\"" | "\\"N\\""

string ::= "\\"" char* "\\""
char ::= [^"\\\\] | "\\\\" ["\\\\/bfnrtu] [0-9a-fA-F]*
ws ::= [ \\t\\n]*
'''.strip()

class ReferenceExtractor:
    def extract_references(self, documents: List[Document]) -> List[str]:
        refs = []
        seen = set()
        for doc in documents:
            c_id = doc.metadata.get('contract_id') or doc.metadata.get('parent_id', 'unknown')
            s_num = doc.metadata.get('section_num', 'N/A')
            s_title = (doc.metadata.get('section_title') or 'Unknown').replace(' ', '_').strip()
            ref = f"{c_id}_Section_{s_num}_{s_title}"
            if ref not in seen:
                refs.append(ref)
                seen.add(ref)
        return refs

def _parse_json_output(raw: str) -> Optional[Dict]:
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(match.group()) if match else json.loads(raw)
    except: return None

def _parse_reasoning(data: Any) -> StructuredReasoning:
    if isinstance(data, dict): return StructuredReasoning.from_dict(data)
    if isinstance(data, str): return StructuredReasoning.from_dict({"steps": [{"statement": data}]})
    return StructuredReasoning([])


def _build_short_to_full_ref_map(valid_refs: List[str]) -> Dict[str, List[str]]:
    short_to_full_refs: Dict[str, List[str]] = {}
    for ref in valid_refs:
        short_id = Citation.get_short_id_from_reference(ref)
        short_to_full_refs.setdefault(short_id, []).append(ref)
    return short_to_full_refs


def collect_validated_citations(
    output_dict: Dict[str, Any],
    valid_refs: List[str],
    citation_mode: Literal["short", "both"] = "short",
) -> List[Citation]:
    """Collect citations deterministically from LLM CoT reasoning steps.

    This is the shared collector for both RDG and prompt-only baselines.
    It maps short IDs used in reasoning steps back to full references from
    the retrieved context (`valid_refs`) in a stable order.

    Args:
        output_dict: Parsed JSON output produced by the model.
        valid_refs: Full references built from retrieved context.
        citation_mode:
                        - "short": accept only short IDs (e.g., doc_0_sec_I); full refs
                            and malformed mixed tokens are preserved as invalid citations.
                        - "both": accept exact short IDs and exact full references
                            (e.g., doc_0_Section_I_TITLE); malformed mixed tokens are still
                            preserved as invalid citations.
    """
    if citation_mode not in ("short", "both"):
        raise ValueError(f"Unsupported citation_mode: {citation_mode}")

    reasoning = _parse_reasoning(output_dict.get('reasoning', ''))
    short_to_full_refs = _build_short_to_full_ref_map(valid_refs)
    valid_full_ref_set = set(valid_refs)

    used_citation_tokens_in_order: List[str] = []
    seen_used_tokens = set()

    def _make_invalid_citation(token: str) -> Citation:
        invalid_contract_id = f"__INVALID__::{token.encode('utf-8').hex()}"
        return Citation(token, "Unknown", "Unknown", invalid_contract_id)

    def _normalize_citation_token(token: str) -> str:
        """Keep a citation token observable even when it is syntactically invalid.

        We do not want to silently drop malformed citations like ``""`` or
        dummy IDs copied from the prompt example, because evaluation should
        count them as hallucinated raw citations instead of treating them as
        missing citations.
        """
        if not isinstance(token, str):
            return ""
        stripped = token.strip()
        return stripped if stripped else "__EMPTY_CITATION__"

    if isinstance(reasoning, StructuredReasoning) and reasoning.steps:
        for step in reasoning.steps:
            for short_id in (step.citations or []):
                if not isinstance(short_id, str):
                    continue
                short_id = _normalize_citation_token(short_id)
                if short_id in seen_used_tokens:
                    continue
                seen_used_tokens.add(short_id)
                used_citation_tokens_in_order.append(short_id)

    validated_citations: List[Citation] = []
    seen_reference_ids = set()
    for cit in used_citation_tokens_in_order:
        matched_citations: List[Citation]
        if cit in short_to_full_refs:
            matched_citations = [Citation.from_reference_string(ref) for ref in short_to_full_refs[cit]]
        elif citation_mode == "both" and cit in valid_full_ref_set:
            matched_citations = [Citation.from_reference_string(cit)]
        else:
            # Preserve invalid citation tokens so downstream evaluation can
            # count them as hallucinated citations instead of collapsing them
            # into a valid article ID.
            matched_citations = [_make_invalid_citation(cit)]

        for citation in matched_citations:
            if citation.reference_id in seen_reference_ids:
                continue
            seen_reference_ids.add(citation.reference_id)
            validated_citations.append(citation)

    return validated_citations

# ==========================================
# MAIN PIPELINE
# ==========================================

def _available_cpus() -> int:
    """CPUs this process may run on: the job's allocation under Slurm, all cores elsewhere."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS
        return os.cpu_count() or 1


class RDGPipeline:
    def __init__(
        self,
        model_path: str = None,
        n_ctx: int = DEFAULT_N_CTX,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        seed: int = 42,
    ):
        if not LLAMA_CPP_AVAILABLE: raise RuntimeError("llama-cpp-python missing")
        
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.seed = int(seed)
        self.n_ctx = int(n_ctx)
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"GGUF Model not found at {self.model_path}")
        
        # llama-cpp's defaults (cpu_count()//2 and cpu_count()) count every core on the machine. Under Slurm
        # that is the whole node, so dozens of threads would share the few CPUs the job was given.
        # Keep the same rule, applied to the CPUs this process may actually use.
        cpus = _available_cpus()
        self.n_threads = max(1, cpus // 2)
        self.n_threads_batch = cpus

        print(f"\n🔧 Loading RDG Model: {os.path.basename(self.model_path)}")
        try:
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
                seed=self.seed,
                n_threads=self.n_threads,
                n_threads_batch=self.n_threads_batch,
            )
        finally:
            sys.stderr = old_stderr

        self.grammar_builder = GBNFGrammarBuilder()
        self.extractor = ReferenceExtractor()
        print(f"✓ Model Loaded (Ctx: {n_ctx}, Seed: {self.seed}, Threads: {self.n_threads}/{self.n_threads_batch})")

    def _get_model_n_ctx(self) -> int:
        """Best-effort retrieval of active model context window size."""
        try:
            ctx = self.llm.n_ctx()
            if isinstance(ctx, int) and ctx > 0:
                return ctx
        except Exception:
            pass

        try:
            ctx = getattr(self.llm, "n_ctx", None)
            if isinstance(ctx, int) and ctx > 0:
                return ctx
        except Exception:
            pass

        return self.n_ctx

    def _count_prompt_tokens(self, prompt: str) -> int:
        """Count prompt tokens with llama tokenizer; fallback to rough approximation."""
        try:
            return len(self.llm.tokenize(prompt.encode("utf-8")))
        except Exception:
            return max(1, len(prompt) // 4)

    def resolve_max_tokens(self, prompt: str, max_tokens: Optional[int], reserve_tokens: int = 64) -> int:
        """Resolve generation budget from model context and prompt length.

        If max_tokens is None or <= 0, use the largest available budget.
        Otherwise, clamp requested max_tokens to the available budget.
        """
        prompt_tokens = self._count_prompt_tokens(prompt)
        model_ctx = self._get_model_n_ctx()
        available = max(1, model_ctx - prompt_tokens - max(0, reserve_tokens))

        if max_tokens is None or max_tokens <= 0:
            return available
        return max(1, min(int(max_tokens), available))

    def _build_prompt(self, question: str, valid_refs: List[str], context_str: str, answer_mode: Literal["freeform", "yn"] = "freeform") -> str:
        refs_display = "\n".join([f"- {r} [{Citation.get_short_id_from_reference(r)}]" for r in valid_refs])

        task_hint = ""
        json_example_answer = "Direct answer synthesizing reasoning..."
        if answer_mode == "yn":
            task_hint = (
                "\nTASK:\n"
                "- Output answer as 'Y' if the statement is entailed/true given the context, otherwise 'N'.\n"
                "- You must choose exactly one of: Y or N.\n"
            )
            json_example_answer = "Y"

        return f"""<|start_header_id|>system<|end_header_id|>

You are a systematic reasoning assistant. Answer based ONLY on the provided context.

## INSTRUCTIONS
**Use step-by-step reasoning**
   - extracted_snippet: COPY exact text from context.
   - deduction: Logical inference based on the snippet.
   - type: premise/inference/conclusion.
   - citations: Source ID.

{task_hint}

## OUTPUT FORMAT
Return a valid JSON object matching this schema:
{{
  "reasoning": {{
    "steps": [
      {{ "extracted_snippet": "Exact quote from context...", "deduction": "This implies X...", "type": "premise", "citations": ["doc_0_sec_I"] }},
      {{ "extracted_snippet": "Another relevant quote...", "deduction": "From X, we deduce Y...", "type": "inference", "citations": ["doc_0_sec_I"] }},
      {{ "extracted_snippet": "Supporting text...", "deduction": "Context states Z...", "type": "premise", "citations": ["doc_1_sec_II"] }},
      {{ "extracted_snippet": "", "deduction": "Combining Y and Z to conclude...", "type": "conclusion", "citations": ["doc_0_sec_I", "doc_1_sec_II"] }}
    ]
  }},
    "answer": "{json_example_answer}"
}}

## NOTES
1.  **Citation:** Every factual claim MUST cite the source using ONLY the Short IDs string (the part inside the brackets [ ]) from the VALID_CITATIONS list.
2.  **Grounding:** Every deduction must be grounded in a verbatim snippet from the CONTEXT.
3.  **Reasoning Types:** Use "premise" for direct facts from context, "inference" for logical deductions, and "conclusion" for synthesis steps.
4.  **Inference:** You can combine multiple reasoning steps to make inferences, but you cannot make unsupported assumptions. If you combine multiple snippets, cite all relevant sources.
5.  **No Outside Knowledge:** Do NOT use external knowledge. If context is insufficient, state this explicitly.

<|eot_id|><|start_header_id|>user<|end_header_id|>

## CONTEXT
{context_str}

## VALID_CITATIONS (format: FULL IDs [SHORT IDs])
{refs_display}

## QUESTION
{question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    def generate(
        self,
        question: str,
        documents: List[Document],
        max_tokens: Optional[int] = None,
        temperature: float = 0,
        answer_mode: Literal["freeform", "yn"] = "freeform",
    ) -> StructuredOutput:
        valid_refs = self.extractor.extract_references(documents)
        print(f"📋 Valid Citations: {len(valid_refs)}")
        
        if not valid_refs:
            return StructuredOutput([], StructuredReasoning([]), "No context available.")

        # 1. Build Grammar
        try:
            grammar_str = self.grammar_builder.build_structured_cot_grammar(valid_refs, answer_mode=answer_mode)
            grammar = LlamaGrammar.from_string(grammar_str)
        except Exception as e:
            # Report a failed output: without metadata the evaluator would count it as parsed JSON
            # with zero citations, i.e. a free 0% fabrication.
            message = f"Grammar Error: {e}"
            return StructuredOutput([], StructuredReasoning([]), message,
                                    metadata={"json_parse_success": 0, "raw_output_text": message})

        # 2. Build Context & Prompt
        context_parts = []
        for doc in documents:
            c_id = doc.metadata.get('contract_id') or doc.metadata.get('parent_id', 'unknown')
            s_num = doc.metadata.get('section_num', 'N/A')
            s_title = (doc.metadata.get('section_title') or 'Unknown').replace(' ', '_').strip()
            ref = f"{c_id}_Section_{s_num}_{s_title}"
            context_parts.append(f"[{ref}]\n{doc.page_content.strip()}")
        
        prompt = self._build_prompt(question, valid_refs, "\n\n---\n\n".join(context_parts), answer_mode=answer_mode)
        effective_max_tokens = self.resolve_max_tokens(prompt, max_tokens=max_tokens)

        # 3. Generate
        print(f"🔄 RDG Thinking... (max_tokens={effective_max_tokens})", end="", flush=True)
        output = self.llm(
            prompt,
            grammar=grammar,
            max_tokens=effective_max_tokens,
            temperature=temperature,
            repeat_penalty=1.12,  # Anti-loop
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )
        
        raw_text = output['choices'][0]['text'].strip()
        print(f"✓ Done ({len(raw_text)} chars)")

        # 4. Parse & Validate
        output_dict = _parse_json_output(raw_text)
        if not output_dict:
            return StructuredOutput(
                citations=[],
                reasoning=StructuredReasoning([]),
                answer=raw_text,
                metadata={
                    "json_parse_success": 0,
                    "raw_output_text": raw_text,
                },
            )

        # Basic validation
        reasoning = _parse_reasoning(output_dict.get('reasoning', ''))
        validated_citations = collect_validated_citations(output_dict, valid_refs)

        return StructuredOutput(
            citations=validated_citations,
            reasoning=reasoning,
            answer=output_dict.get('answer', ''),
            metadata={
                "json_parse_success": 1,
                "raw_output_text": raw_text,
                "effective_max_tokens": effective_max_tokens,
            },
        )

# ==========================================
# SINGLETON EXPORT
# ==========================================
_rdg_instances: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], RDGPipeline] = {}


def _to_hashable(value: Any) -> Any:
    """Best-effort conversion for cache key stability."""
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _build_rdg_cache_key(model_path: Optional[str], kwargs: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
    resolved_model_path = os.path.abspath(model_path or DEFAULT_MODEL_PATH)
    normalized_kwargs = tuple(sorted((k, _to_hashable(v)) for k, v in kwargs.items()))
    return resolved_model_path, normalized_kwargs


def get_rdg_pipeline(model_path: str = None, **kwargs):
    key = _build_rdg_cache_key(model_path, kwargs)
    if key in _rdg_instances:
        return _rdg_instances[key]
    _rdg_instances[key] = RDGPipeline(model_path=model_path, **kwargs)
    return _rdg_instances[key]

def get_rdg_mode() -> str:
    return 'rdg' if _rdg_instances else 'unknown'
