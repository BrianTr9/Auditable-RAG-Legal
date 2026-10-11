"""
Semantic-Preserving Hierarchical Retrieval (SPHR) - Layer 1
==========================================================

Hierarchical chunking with parent-child structure for section-aware retrieval.

Approach:
- Parents: Full sections (typically 1,000-3,000 tokens, split at semantic boundaries if oversized)
- Children: ~300-token chunks with ~90-token overlap
- Semantic boundaries: Preserves paragraph/sentence integrity
- Retrieval: Expands top-k children to their parent sections
- Cross-Reference Enrichment: Resolves §X.X.X references to actual section content

Author: Trung Bao (Brian) Truong
Date: January 2026
"""

import re
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from rank_bm25 import BM25Okapi

try:
    import tiktoken
except ImportError:
    raise ImportError(
        "tiktoken is required for SPHR token-based chunking. "
        "Install dependencies first (e.g., pip install -r requirements.txt)."
    )


@dataclass
class Chunk:
    """
    Represents a hierarchical chunk with metadata
    
    Attributes:
        text: The actual text content
        chunk_id: Unique identifier (e.g., "parent_2.1" or "parent_2.1_child_0")
        chunk_type: Either "parent" or "child"
        parent_id: Parent chunk ID (None for parents, required for children)
        section_num: Section number from contract (e.g., "2.1")
        section_title: Section title (e.g., "OVERVIEW")
        token_count: Number of tokens in this chunk
        char_start: Starting character position in original document
        char_end: Ending character position in original document
        contract_id: Identifier of source contract
        provision: True for heading-delimited provisions (and their parts); False for the
            preamble and the whole-document fallback, which are not cross-reference targets
    """
    text: str
    chunk_id: str
    chunk_type: str  # "parent" or "child"
    parent_id: str = None
    section_num: str = None
    section_title: str = None
    token_count: int = 0
    char_start: int = 0
    char_end: int = 0
    contract_id: str = None
    provision: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class HierarchicalChunker:
    """
    General hierarchical chunking system with semantic boundary detection
    
    Design Philosophy:
    1. Parents = Full sections (complete context)
    2. Children = Small chunks (~300 tokens, precise retrieval)
    3. Semantic boundaries (paragraph/sentence/clause aware)
    4. Retrieval: Search children → expand to parents
    5. Metadata-rich chunks for auditing and analysis
    """
    
    # Section detection patterns (priority order: Japanese statute formats → ARTICLE/Part/Section variants → bare Roman/decimal)
    # Focuses on formal headings with explicit separators (-, :, .) or space
    # TOC EXCLUSION: Excludes TOC entries with pattern "3+ dots + optional spaces + digits"
    # Strategy: Use negative lookahead (?!.*\.{3,}\s*\d+$) to prevent matching TOC entries
    # E.g., "1. DEFINITIONS" matches, but "1. DEFINITIONS ..................... 1" does NOT
    # But allows ellipsis in headers like "1. TITLE ... SOMETHING" (dots not followed by number)
    SECTION_PATTERNS = [
    # 0. JAPANESE STATUTE ARTICLES (COLIEE/e-Gov style)
    # Matches (optionally preceded by a caption line):
    #   （未成年者の法律行為）\n第五条　...
    #   第三条の二　...
    #   第３条　... (full-width digits)
    # Captures:
    #   section: e.g., "第五条", "第三条の二" (normalized later to "5", "3-2")
    #   title: optional caption inside （） on the previous line
    (r'^(?:\uff08(?P<title>[^\uff09\n]+)\uff09\s*\n)?(?P<section>\u7b2c[0-9\uff10-\uff19\u3007\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343]+\u6761(?:\u306e[0-9\uff10-\uff19\u3007\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343]+)*)[\u3000 \t]+', 'Japanese Article Format'),

    # 1. KEYWORD FORMATS (Specific keyword -> Safe to put first)
    # Matches: ARTICLE I - Title, Part 1: Title...
    # Allow optional same-line title (may be redacted) so headers like "ARTICLE I." match
    # Anchored at line start (indentation allowed) so that "... set forth in Section 2." in
    # running text is a reference, not a heading.
    (r'^[ \t]*(?:ARTICLE|Article|PART|Part|SECTION|Section)\s+([IVX]+|\d+)\s*(?:[-:.]\s*|\s+)(?:\*+(?:\s+\*+)*\s*)?([A-Z](?:(?!\.{3,}\s*\d)[^\n])*?)?(?=\n|$)', 'Keyword (Article/Part/Section) Format'),

    # 2. NESTED DECIMALS (Must be before Bare Decimals to avoid partial match)
    # Matches: 1.1, 1.1.1, 2.1.3...
    (r'^(\d+(?:\.\d+)+)\s*(?:[-:.]\s*|\s+)([A-Z](?:(?!\.{3,}\s*\d)[^\n])*?)(?=\n|$)', 'Nested Decimal Format'),

    # 3. BARE ROMAN NUMERALS
    # Matches: I, II, IV...
    (r'^([IVX]+)\s*(?:[-:.]\s*|\s+)([A-Z](?:(?!\.{3,}\s*\d)[^\n])*?)(?=\n|$)', 'Bare Roman Format'),

    # 4. BARE DECIMALS (Most general)
    # Matches: 1, 2, 10...
    (r'^(\d+)\s*(?:[-:.]\s*|\s+)([A-Z](?:(?!\.{3,}\s*\d)[^\n])*?)(?=\n|$)', 'Bare Digit Format'),
    
    # 5. BARE SINGLE LETTER
    # Matches: A, B, C ... (treated like top-level section markers)
    (r'^([A-Z])\s*(?:[-:.]\s*|\s+)([A-Z](?:(?!\.{3,}\s*\d)[^\n])*?)(?=\n|$)', 'Bare Letter Format'),
]
    
    # Cross-reference patterns: detect inline mentions of sections (e.g., "Section 1", "Article II")
    # Used in body text to find references to other sections for graph enrichment
    # All patterns use re.IGNORECASE flag in enrich_parent_to_children_with_cross_refs()
    # Pattern 1 catches all keyword-based references (including contextual like "see Section 1" and positional like "Section 1 above")
    # Pattern 2 specifically handles parenthetical/bracketed forms which have different structure
    CROSS_REF_PATTERNS = [
        # 1. Keyword-based references: "Section 1", "Article II", "Part 2.1", "subsection 1.2", "clause 3", "§3"
        # Also matches: "see Section 1", "refer to Article II", "Section 1 above", "as described in Part 2"
        # Matches: Section/Article/Part/Subsection/Clause/Paragraph + section number
        # Also matches abbreviations: Sec./Art./Para. (with dot) and symbols: §/¶ (with or without space)
        # Roman numerals must end at a word boundary: under IGNORECASE, "clause is" / "part in"
        # would otherwise capture "i" as Section I.
        r'(?:section|article|part|subsection|clause|paragraph|sec\.|art\.|para\.)\s+([IVX]+\b|\d+(?:\.\d+)*)|(?:§|¶)\s*([IVX]+\b|\d+(?:\.\d+)*)',
        
        # 2. Parenthetical/bracketed references: "(Section 1)", "[Article II]", "(see §3)"
        # Matches: references enclosed in parentheses or brackets (symbols §/¶ with or without space)
        r'[\(\[](?:see\s+)?(?:section|article|part)\s+([IVX]+|\d+(?:\.\d+)*)[\)\]]|[\(\[](?:see\s+)?(?:§|¶)\s*([IVX]+|\d+(?:\.\d+)*)[\)\]]',

        # 3. Japanese statute article references inside text: "第七条", "第三条の二"
        # Captures the full identifier so it can be normalized to match section_num (e.g., "第三条の二" -> "3-2").
        r'(\u7b2c[0-9\uff10-\uff19\u3007\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343]+\u6761(?:\u306e[0-9\uff10-\uff19\u3007\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343]+)*)',
    ]
    
    # Separator hierarchy (paragraph → sentence → line → clause → word)
    # Japanese legal text (e.g., COLIEE civil code) is primarily segmented by:
    #   - Paragraph breaks: \n\n
    #   - Sentence terminator: 。 (and full-width variants)
    #   - Clause separators: 、 ： ；
    SEPARATORS = [
        ('\n\n', 'paragraph'),
        ('。\n', 'sentence_line_jp'),
        ('。', 'sentence_jp'),
        ('．\n', 'sentence_line_fullwidth'),
        ('． ', 'sentence_fullwidth_space'),
        ('．', 'sentence_fullwidth'),
        ('.\n', 'sentence_line'),
        ('. ', 'sentence'),
        ('.', 'sentence'),
        ('\n', 'line'),
        ('； ', 'semicolon_fullwidth_space'),
        ('；', 'semicolon_fullwidth'),
        ('; ', 'semicolon_space'),
        (';', 'semicolon'),
        ('： ', 'colon_fullwidth_space'),
        ('：', 'colon_fullwidth'),
        (': ', 'colon_space'),
        (':', 'colon'),
        ('、 ', 'comma_jp_space'),
        ('、', 'comma_jp'),
        (', ', 'comma_space'),
        (',', 'comma'),
        ('\u3000', 'ideographic_space'),
        (' ', 'word'),
    ]

    @staticmethod
    def _normalize_fullwidth_digits(text: str) -> str:
        """Convert full-width digits (０-９) to ASCII digits (0-9)."""
        if not text:
            return text
        fw = '０１２３４５６７８９'
        hw = '0123456789'
        return text.translate(str.maketrans(fw, hw))

    @staticmethod
    def _roman_to_arabic(roman: str) -> int:
        """Convert Roman numeral (I, II, III, IV, V, X, etc.) to Arabic integer.

        Returns -1 for invalid input.
        """
        if not roman or not isinstance(roman, str):
            return -1
        roman = roman.strip().upper()
        vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        total = 0
        prev = 0
        for ch in reversed(roman):
            if ch not in vals:
                return -1
            cur = vals[ch]
            if cur < prev:
                total -= cur
            else:
                total += cur
            prev = cur
        return total if total > 0 else -1

    @staticmethod
    def _arabic_to_roman(num: int) -> str:
        """Convert Arabic integer (1-3999) to Roman numeral string. Returns empty string if out of range."""
        if not isinstance(num, int) or num <= 0 or num >= 4000:
            return ""
        pairs = [
            (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
            (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
            (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
        ]
        out = []
        for val, sym in pairs:
            count = num // val
            if count:
                out.append(sym * count)
                num -= val * count
        return ''.join(out)

    @staticmethod
    def _kanji_numeral_to_int(text: str) -> int:
        """Convert common Japanese kanji numerals to int (supports up to 万)."""
        if not text:
            raise ValueError("empty kanji numeral")

        text = HierarchicalChunker._normalize_fullwidth_digits(text)
        if text.isdigit():
            return int(text)

        digit_map = {
            '〇': 0, '零': 0,
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9,
        }
        unit_map = {'十': 10, '百': 100, '千': 1000, '万': 10000}

        total = 0
        section_total = 0
        current = 0

        for ch in text:
            if ch in digit_map:
                current = digit_map[ch]
                continue

            if ch in unit_map:
                unit = unit_map[ch]
                if current == 0:
                    current = 1
                if unit == 10000:
                    section_total = (section_total + current) * unit
                    total += section_total
                    section_total = 0
                else:
                    section_total += current * unit
                current = 0
                continue

            raise ValueError(f"unsupported kanji numeral char: {ch}")

        return total + section_total + current

    @staticmethod
    def _normalize_japanese_article_identifier(section_raw: str) -> str:
        """Normalize Japanese article identifiers like '第五条', '第三条の二' -> '5', '3-2'."""
        if not section_raw:
            return section_raw

        section_raw = HierarchicalChunker._normalize_fullwidth_digits(section_raw)
        section_raw = section_raw.strip()

        # Expect patterns like: 第...条 or 第...条の...
        if not section_raw.startswith('第') or '条' not in section_raw:
            return section_raw

        base_part = section_raw[1:section_raw.index('条')]
        suffix_part = section_raw[section_raw.index('条') + 1:]

        def parse_part(part: str) -> str:
            part = part.strip()
            if not part:
                return ''
            if part.isdigit():
                return str(int(part))
            try:
                return str(HierarchicalChunker._kanji_numeral_to_int(part))
            except Exception:
                return part

        base_norm = parse_part(base_part)
        if not suffix_part:
            return base_norm

        # suffix like: の二, の２, の十五 ... possibly repeated
        suffixes = []
        for piece in suffix_part.split('の'):
            if not piece:
                continue
            suffixes.append(parse_part(piece))

        if suffixes:
            return base_norm + '-' + '-'.join(suffixes)
        return base_norm
    
    def __init__(
        self,
        child_size: int = 300,
        child_overlap: int = 90,
        encoding_name: str = "cl100k_base",
        semantic_boundary_tolerance: int = 30,
        parent_max_tokens: int = 3000,
        parent_overlap_ratio: float = 0.1,
        parent_split_tolerance: int = 1000
    ):
        """
        Initialize chunker with validation
        
        Args:
            child_size: Target size for child chunks in tokens (default: 300)
            child_overlap: Overlap between consecutive children in tokens (default: 90 = 30%)
            encoding_name: Tiktoken encoding (default: cl100k_base for GPT-4/Gemma)
            semantic_boundary_tolerance: ±tokens to search for boundaries (default: 30 = ±10%)
        """
        self._validate_config(child_size, child_overlap, semantic_boundary_tolerance)
        
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.semantic_boundary_tolerance = semantic_boundary_tolerance
        # Maximum tokens allowed per parent chunk before splitting
        self.parent_max_tokens = parent_max_tokens
        # Parent overlap (in tokens) when splitting parent into parts
        # If ratio provided, compute tokens from ratio of parent_max_tokens
        self.parent_overlap_ratio = parent_overlap_ratio
        self.parent_overlap_tokens = max(0, int(self.parent_max_tokens * self.parent_overlap_ratio))
        # Tolerance (in tokens) when searching for semantic boundary for parent splits
        self.parent_split_tolerance = parent_split_tolerance
        self.tokenizer = tiktoken.get_encoding(encoding_name)
    
    def _validate_config(self, child_size: int, child_overlap: int, semantic_boundary_tolerance: int):
        """Validate initialization parameters"""
        if child_size <= 0:
            raise ValueError("child_size must be positive")
        if child_overlap < 0:
            raise ValueError("child_overlap cannot be negative")
        if child_overlap >= child_size:
            raise ValueError("child_overlap must be less than child_size")
        if semantic_boundary_tolerance <= 0:
            raise ValueError("semantic_boundary_tolerance must be positive")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))
    
    def extract_sections(self, contract_text: str) -> List[Dict]:
        """
        Extract heading-delimited provisions from a structured document

        Headings start a line, e.g. "ARTICLE II - PAYMENT", "1.1 Provision of Services.",
        "2. Compensation.", "第五条　...". Subsections (1.1, 1.2) are grouped into their major
        section (1). The returned sections partition the document: a preamble ('0') holds any
        text before the first provision, and a document without headings is one parent
        ('1', 'Full Document'). See _detect_headings and _provision_starts for the rules.

        Args:
            contract_text: Text to extract sections from
            
        Returns:
            List of section dictionaries with metadata
            
        Raises:
            ValueError: If contract_text is empty or None
        """
        # Validate input
        if not contract_text or not isinstance(contract_text, str):
            raise ValueError("contract_text must be non-empty string")
        
        contract_text = contract_text.strip()
        if not contract_text:
            raise ValueError("contract_text is empty after stripping")
        
        headings = self._detect_headings(contract_text)
        if not headings:
            print("⚠️  No structured sections found, using entire document as one parent")
            return [self._make_section(contract_text, 0, len(contract_text), '1', 'Full Document', provision=False)]

        starts = self._provision_starts(headings, len(contract_text))

        result_sections = []
        # Text before the first provision (title, recitals, table of contents) is kept as a
        # preamble parent (section '0') so no part of the document is dropped.
        preamble = self._make_section(contract_text, 0, starts[0][0], '0', 'Preamble', provision=False)
        if preamble:
            result_sections.append(preamble)

        ends = [start for start, _, _ in starts[1:]] + [len(contract_text)]
        for (start, major_num, title), end in zip(starts, ends):
            section = self._make_section(contract_text, start, end, major_num, title, provision=True)
            if section:
                result_sections.append(section)
            else:
                print(f"  ⊘ Skipping empty Section {major_num}")

        return result_sections

    def _make_section(self, text: str, start: int, end: int, section_num: str, title: str,
                      provision: bool) -> Dict:
        """Section dict for text[start:end], whitespace-trimmed with matching char offsets."""
        raw = text[start:end]
        stripped = raw.strip()
        if not stripped:
            return None
        char_start = start + len(raw) - len(raw.lstrip())
        return {
            'section_num': section_num,
            'title': title,
            'text': stripped,
            'char_start': char_start,
            'char_end': char_start + len(stripped),
            'token_count': self.count_tokens(stripped),
            'provision': provision,
        }

    def _match_section_num(self, match) -> str:
        groupdict = match.groupdict()
        section_num = (groupdict.get('section') or match.group(1) or '').strip()
        if section_num.startswith('第') and '条' in section_num:
            section_num = self._normalize_japanese_article_identifier(section_num)
        return section_num

    @staticmethod
    def _match_title(match) -> str:
        # Title may come from a named group (e.g., Japanese caption line) or the first
        # non-empty captured group after the section number.
        title = (match.groupdict().get('title') or "").strip()
        if not title:
            for group_idx in range(2, len(match.groups()) + 1):
                val = match.group(group_idx)
                if val:
                    title = val.strip()
                    break
        return title

    def _detect_headings(self, text: str) -> List[Tuple[int, str, str]]:
        """Headings (start, section_num, title) in document order, or [] if none delimit the text.

        Patterns are tried in priority order. A pattern is accepted only if its headings can
        delimit provisions: they name at least two distinct major sections, or the single
        provision they name starts at the beginning of the text (a corpus unit that is itself
        one provision, e.g. a statute article). One stray line that happens to look like a
        heading therefore cannot override the structure found by a later pattern.
        """
        for pattern_idx, (pattern, pattern_name) in enumerate(self.SECTION_PATTERNS):
            headings = [
                (m.start(), self._match_section_num(m), self._match_title(m))
                for m in re.finditer(pattern, text, re.MULTILINE)
            ]
            if not headings:
                continue
            majors = {num.split('.')[0] for _, num, _ in headings}
            if len(majors) >= 2 or headings[0][0] == 0:
                print(f"📋 Found {len(headings)} sections using pattern {pattern_idx + 1} ({pattern_name})")
                return headings
        return []

    @staticmethod
    def _provision_starts(headings: List[Tuple[int, str, str]], text_len: int) -> List[Tuple[int, str, str]]:
        """Reduce headings to one start (start, major_num, title) per provision.

        1. Consecutive headings with the same major number open one provision
           (1.1, 1.2, 1.3 -> Section 1, titled by its first heading).
        2. A major number that opens several provisions (a table of contents listing the
           headings, or numbering restarted later) keeps the occurrence with the longest span;
           the other occurrences stay inside the parent that precedes them. Parents therefore
           keep unique IDs and still partition the document.
        """
        starts = []
        for start, section_num, title in headings:
            major_num = section_num.split('.')[0]
            if starts and starts[-1][1] == major_num:
                continue
            starts.append((start, major_num, title))

        ends = [start for start, _, _ in starts[1:]] + [text_len]
        longest = {}
        for (start, major_num, _), end in zip(starts, ends):
            if major_num not in longest or end - start > longest[major_num][0]:
                longest[major_num] = (end - start, start)
        return [s for s in starts if longest[s[1]][1] == s[0]]
    
    @staticmethod
    def _section_to_parent_chunk(sec: Dict, contract_id: str) -> Chunk:
        """Convert section dict to parent Chunk object"""
        return Chunk(
            text=sec['text'],
            chunk_id=f"{contract_id}_parent_{sec['section_num']}",
            chunk_type="parent",
            parent_id=None,
            section_num=sec['section_num'],
            section_title=sec['title'],
            token_count=sec['token_count'],
            char_start=sec['char_start'],
            char_end=sec['char_end'],
            contract_id=contract_id,
            provision=sec.get('provision', True),
        )
    
    def create_parent_chunks(self, sections: List[Dict], contract_id: str = "contract") -> List[Chunk]:
        """Create parent chunks from sections (one chunk per section)"""
        return [self._section_to_parent_chunk(sec, contract_id) for sec in sections]

    def _split_oversized_parents(self, parents: List[Chunk], contract_id: str) -> List[Chunk]:
        """
        Split parent chunks that exceed `parent_max_tokens + semantic_boundary_tolerance`.

        Returns a new list of parents where oversized parents are replaced by smaller parent parts.
        """
        new_parents = []
        for parent in parents:
            threshold = self.parent_max_tokens + self.parent_split_tolerance
            if parent.token_count <= threshold:
                new_parents.append(parent)
                continue

            # Tokenize parent text and build token->char mapping
            parent_tokens = self.tokenizer.encode(parent.text)
            token_to_char = {}
            for pos in range(len(parent_tokens) + 1):
                token_to_char[pos] = len(self.tokenizer.decode(parent_tokens[:pos]))

            tokens_len = len(parent_tokens)
            step = max(1, self.parent_max_tokens - self.parent_overlap_tokens)

            part_idx = 0
            start = 0
            while start < tokens_len:
                target_end = min(start + self.parent_max_tokens, tokens_len)

                # If not at doc end, find semantic boundary near target_end using parent_split_tolerance
                if target_end < tokens_len:
                    actual_end = self.find_semantic_boundary(
                        parent_tokens,
                        target_end,
                        tolerance=self.parent_split_tolerance
                    )
                    # Prevent zero-length or backwards moves
                    if actual_end <= start:
                        actual_end = target_end
                else:
                    actual_end = target_end

                # Extract token slice and compute char offsets
                chunk_tokens = parent_tokens[start:actual_end]
                chunk_text = self.tokenizer.decode(chunk_tokens).strip()
                token_count = len(chunk_tokens)

                char_start_in_parent = token_to_char[start]
                char_end_in_parent = token_to_char[actual_end]
                char_start = parent.char_start + char_start_in_parent
                char_end = parent.char_start + char_end_in_parent

                part_idx += 1
                new_chunk_id = f"{contract_id}_parent_{parent.section_num}_part_{part_idx}"

                new_parent = Chunk(
                    text=chunk_text,
                    chunk_id=new_chunk_id,
                    chunk_type="parent",
                    parent_id=None,
                    section_num=f"{parent.section_num}_part_{part_idx}",
                    section_title=parent.section_title,
                    token_count=token_count,
                    char_start=char_start,
                    char_end=char_end,
                    contract_id=parent.contract_id,
                    provision=parent.provision,
                )
                new_parents.append(new_parent)

                # Advance start with overlap
                if actual_end >= tokens_len:
                    break
                start = max(0, actual_end - self.parent_overlap_tokens)

        return new_parents
    
    def split_by_separators(self, text: str, target_size: int) -> List[str]:
        """
        Split text using separator priority hierarchy
        
        Priority: \\n\\n → .\\s → \\n → ;\\s → ,\\s → space
        
        Args:
            text: Text to split
            target_size: Target size in tokens
            
        Returns:
            List of text chunks
        """
        if self.count_tokens(text) <= target_size:
            return [text]
        
        # Try each separator in priority order (DRY: use class constant)
        for separator, sep_name in self.SEPARATORS:
            parts = text.split(separator)
            
            if len(parts) <= 1:
                # This separator doesn't exist, try next
                continue
            
            # Rebuild chunks respecting target size
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            for i, part in enumerate(parts):
                # Add separator back (except for last part)
                if i < len(parts) - 1:
                    part_with_sep = part + separator
                else:
                    part_with_sep = part
                
                part_tokens = self.count_tokens(part_with_sep)
                
                # Check if adding this part exceeds target
                if current_tokens + part_tokens <= target_size or not current_chunk:
                    current_chunk.append(part_with_sep)
                    current_tokens += part_tokens
                else:
                    # Save current chunk and start new one
                    chunks.append(''.join(current_chunk))
                    current_chunk = [part_with_sep]
                    current_tokens = part_tokens
            
            # Save last chunk
            if current_chunk:
                chunks.append(''.join(current_chunk))
            
            # Check if splitting was successful
            max_chunk_size = max(self.count_tokens(c) for c in chunks)
            if max_chunk_size <= target_size * 1.2:  # Allow 20% tolerance
                return [c.strip() for c in chunks if c.strip()]
        
        # If all separators failed, force split by tokens
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), target_size):
            chunk_tokens = tokens[i:i + target_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        
        return chunks
    
    def find_semantic_boundary(
        self,
        tokens: list,
        target_pos: int,
        tolerance: int = None
    ) -> int:
        """
        Find semantic boundary near target position
        
        Priority: paragraph → sentence → line → semicolon → comma
        
        Args:
            tokens: List of tokens
            target_pos: Target position to find boundary near
            tolerance: Search range ±tolerance from target. If None, uses self.semantic_boundary_tolerance
            
        Returns:
            Best boundary position, or target_pos if none found
        """
        if tolerance is None:
            tolerance = self.semantic_boundary_tolerance
        # Search range
        start_search = max(0, target_pos - tolerance)
        end_search = min(len(tokens), target_pos + tolerance)
        
        if start_search >= end_search:
            return target_pos
        
        # Decode search window to text
        search_tokens = tokens[start_search:end_search]
        search_text = self.tokenizer.decode(search_tokens)
        
        # Use class constant separators (DRY), limit to most relevant ones
        # (exclude last one: word - too fine-grained for boundaries)
        boundary_separators = self.SEPARATORS[:-1]
        
        best_boundary = None
        best_priority = -1
        
        for priority, (sep, _) in enumerate(boundary_separators):
            # Find separator by checking each token position directly
            # Avoids encode/decode mismatch by decoding from token list
            best_sep_pos = None
            best_sep_distance = float('inf')
            
            for token_pos in range(start_search, end_search):
                # Decode tokens from BEGINNING to current position
                # (not from start_search - that loses context)
                decoded = self.tokenizer.decode(tokens[:token_pos])
                
                # Check if decoded text ends with this separator
                if decoded.endswith(sep):
                    distance = abs(token_pos - target_pos)
                    if distance < best_sep_distance:
                        best_sep_distance = distance
                        best_sep_pos = token_pos
            
            if best_sep_pos is None:
                continue
            
            # Check if this separator is better than previous (higher priority)
            if best_boundary is None or priority < best_priority:
                best_boundary = best_sep_pos
                best_priority = priority
                
                # If we found paragraph break, that's best possible
                if priority == 0:
                    break
        
        # Return best boundary or target if none found
        return best_boundary if best_boundary is not None else target_pos
    
    def create_child_chunks(self, parent: Chunk) -> List[Chunk]:
        """Create child chunks from parent with sliding window overlap and semantic boundaries"""
        # Validate input
        if not parent.text or not isinstance(parent.text, str):
            raise ValueError(f"Parent {parent.chunk_id} has empty text")
        
        # Tokenize entire parent text
        parent_tokens = self.tokenizer.encode(parent.text)
        if len(parent_tokens) == 0:
            return []
        
        # Pre-calculate token-to-char position mapping for accurate char positions ✓
        # Maps token position -> character position in parent.text
        token_to_char = {}
        for pos in range(len(parent_tokens) + 1):
            chunk_tokens = parent_tokens[:pos]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            token_to_char[pos] = len(chunk_text)
        
        # Create child chunks with overlap
        children = []
        chunk_idx = 0
        position = 0
        
        while position < len(parent_tokens):
            # Target end position
            target_end = min(position + self.child_size, len(parent_tokens))
            
            # Find semantic boundary near target (only if not at end)
            if target_end < len(parent_tokens):
                actual_end = self.find_semantic_boundary(
                    parent_tokens, 
                    target_end,
                    tolerance=self.semantic_boundary_tolerance
                )
            else:
                actual_end = target_end
            
            # Extract and decode chunk
            chunk_tokens = parent_tokens[position:actual_end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            token_count = len(chunk_tokens)
            
            # Validate child size (allow 20% tolerance) ✓
            max_allowed_tokens = int(self.child_size * 1.2)
            if token_count > max_allowed_tokens:
                print(f"⚠️  Child {chunk_idx} exceeds size limit: {token_count} > {max_allowed_tokens} tokens")
            
            # Calculate accurate char positions using token-to-char mapping ✓
            char_start_in_parent = token_to_char[position]
            char_end_in_parent = token_to_char[actual_end]
            chunk_char_start = parent.char_start + char_start_in_parent
            chunk_char_end = parent.char_start + char_end_in_parent
            
            # Create child chunk
            child = Chunk(
                text=chunk_text,
                chunk_id=f"{parent.chunk_id}_child_{chunk_idx}",
                chunk_type="child",
                parent_id=parent.chunk_id,
                section_num=parent.section_num,
                section_title=parent.section_title,
                token_count=token_count,
                char_start=chunk_char_start,
                char_end=chunk_char_end,
                contract_id=parent.contract_id
            )
            children.append(child)
            
            # Move forward with overlap ✓ (Fixed: was max(actual_end - overlap, actual_end))
            # Next chunk starts at (actual_end - overlap) to maintain 30% overlap
            position = actual_end - self.child_overlap
            
            # Prevent infinite loop if position doesn't advance
            if position <= 0 or position >= actual_end:
                position = actual_end
            
            chunk_idx += 1
            
            # Stop if we've covered the entire text
            if actual_end >= len(parent_tokens):
                break
        
        return children
    
    def chunk_contract(
        self,
        contract_text: str,
        contract_id: str = "contract_001",
        output_path: str = None
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Main chunking pipeline with optional save
        
        Args:
            contract_text: Full contract text
            contract_id: Identifier for this contract
            output_path: Optional path to save chunks as JSON
            
        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        print(f"\n{'='*60}")
        print(f"🏗️  Chunking contract: {contract_id}")
        print(f"{'='*60}")
        
        # Step 1: Extract sections
        sections = self.extract_sections(contract_text)
        print(f"📄 Extracted {len(sections)} sections")
        
        # Step 2: Create parent chunks (full sections)
        parents = self.create_parent_chunks(sections, contract_id)
        print(f"👨 Created {len(parents)} parent chunks (before splitting)")

        # Step 2b: Split oversized parents into smaller parent parts if needed
        parents = self._split_oversized_parents(parents, contract_id)
        print(f"👨 Created {len(parents)} parent chunks (after splitting)")
        
        # Step 3: Create child chunks with overlap
        all_children = []
        for parent in parents:
            children = self.create_child_chunks(parent)
            all_children.extend(children)
        
        print(f"👶 Created {len(all_children)} child chunks")
        
        # Statistics
        if parents:
            parent_tokens = [p.token_count for p in parents]
            print(f"\n📊 Parent Statistics:")
            print(f"   Avg: {sum(parent_tokens)/len(parent_tokens):.0f} tokens")
            print(f"   Min: {min(parent_tokens)} tokens")
            print(f"   Max: {max(parent_tokens)} tokens")
        
        if all_children:
            child_tokens = [c.token_count for c in all_children]
            print(f"\n📊 Child Statistics:")
            print(f"   Avg: {sum(child_tokens)/len(child_tokens):.0f} tokens")
            print(f"   Min: {min(child_tokens)} tokens")
            print(f"   Max: {max(child_tokens)} tokens")
        
        # Save if path provided
        if output_path:
            self.save_chunks(parents, all_children, output_path)
        
        return parents, all_children
    
    def save_chunks(self, parents: List[Chunk], children: List[Chunk], output_path: str):
        """Save chunks to JSON file with full metadata"""
        data = {
            'metadata': {
                'child_size': self.child_size,
                'child_overlap': self.child_overlap,
                'num_parents': len(parents),
                'num_children': len(children)
            },
            'parents': [p.to_dict() for p in parents],
            'children': [c.to_dict() for c in children]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved chunks to: {output_path}")


    @staticmethod
    def build_parent_child_mapping(parents: List['Chunk'], children: List['Chunk']) -> Dict[str, List[str]]:
        """
        Build mapping from parent chunk IDs to their child chunk IDs
        
        Args:
            parents: List of parent chunks
            children: List of child chunks
            
        Returns:
            Dictionary mapping parent_id -> [child_ids]
        """
        parent_to_children = {}
        for parent in parents:
            parent_to_children[parent.chunk_id] = [
                child.chunk_id for child in children 
                if child.parent_id == parent.chunk_id
            ]
        return parent_to_children

    def enrich_parent_to_children_with_cross_refs(
        self,
        parent_to_children: Dict[str, List[str]],
        parents: List['Chunk'],
        children: List['Chunk']
    ) -> Dict[str, List[str]]:
        """
        Enrich parent_to_children mapping by detecting cross-references in child text.
        
        When a child mentions another section (e.g., "See Section 2" inside Section 1),
        add that child to the referenced parent's children list as well, creating a
        lightweight graph: retrieving the child then expands to {host} U refs.
        
        Links are document-scoped and only target provisions (heading-delimited parents).
        A reference to a provision that was split into parts links every part, so the
        referenced provision is returned whole. The preamble and the whole-document
        fallback are never targets.
        
        Uses CROSS_REF_PATTERNS class constant for inline section detection.
        
        Args:
            parent_to_children: Initial parent_id -> [child_ids] mapping
            parents: List of parent chunks
            children: List of child chunks
            
        Returns:
            Enriched parent_to_children mapping with cross-reference links
        """
        # contract_id -> {section_num: [parent_id, ...]} (all parts of a split provision)
        targets_by_doc: Dict[str, Dict[str, List[str]]] = {}
        for parent in parents:
            if not parent.provision:
                continue
            base_num = parent.section_num.split('_part_')[0]
            targets_by_doc.setdefault(parent.contract_id, {}).setdefault(base_num, []).append(parent.chunk_id)

        for child in children:
            targets = targets_by_doc.get(child.contract_id)
            if not targets or not child.text:
                continue
            for sec_num in self._find_cross_refs(child.text):
                for parent_id in self._resolve_cross_ref(sec_num, targets):
                    linked = parent_to_children.setdefault(parent_id, [])
                    if child.chunk_id not in linked:
                        linked.append(child.chunk_id)

        return parent_to_children

    def _find_cross_refs(self, text: str) -> List[str]:
        """Section numbers referenced in text, in order of first mention."""
        found = []
        for pattern in self.CROSS_REF_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Patterns use alternation, so take the first non-empty group.
                sec_num = next((g for g in match.groups() if g), None)
                if not sec_num:
                    continue
                if sec_num.startswith('第') and '条' in sec_num:
                    sec_num = self._normalize_japanese_article_identifier(sec_num)
                if sec_num not in found:
                    found.append(sec_num)
        return found

    def _resolve_cross_ref(self, sec_num: str, targets: Dict[str, List[str]]) -> List[str]:
        """Parent IDs for a referenced section number within one document.

        Tries the number as written, then its Roman/Arabic counterpart ('5' <-> 'V'),
        then the same for its major section when the reference names a subsection ('1.2' -> '1').
        """
        candidates = [sec_num.strip()]
        if '.' in candidates[0]:
            candidates.append(candidates[0].split('.')[0])
        for num in candidates:
            if num.isdigit():
                alternate = self._arabic_to_roman(int(num))
            else:
                arabic = self._roman_to_arabic(num)
                alternate = str(arabic) if arabic > 0 else None
            for key in (num, alternate):
                if key and key in targets:
                    return targets[key]
        return []


# Demo utilities were removed from the library module to keep the file
# focused on core chunking and retrieval classes. Use a separate script
# for ad-hoc demonstrations or debugging (e.g., tools/demo_sphr.py).


# ==========================================
# LAYER 1: RETRIEVAL COMPONENTS
# ==========================================

class HierarchicalRetriever:
    """
    Retriever that implements hierarchical retrieval strategy:
    1. Search for child chunks (high precision)
    2. Expand to parent chunks (full context)
    
    This is the retrieval component of Layer 1.
    """
    
    def __init__(
        self, 
        vectordb,
        parent_chunks: List['Chunk'], 
        parent_to_children: Dict[str, List[str]],
        top_k: int = 4,
        context_budget: int = 3000,
        retrieval_mode: str = "dense",
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        """
        Initialize hierarchical retriever
        
        Args:
            vectordb: ChromaDB or similar vector database
            parent_chunks: List of parent Chunk objects
            parent_to_children: Mapping of parent_id → [child_ids]
            top_k: Number of child chunks to retrieve
            context_budget: Maximum tokens allowed in context (default 3000, leaving 1K for generation in 4K window)
            retrieval_mode: Retrieval strategy: "dense", "bm25", or "hybrid"
            bm25_weight: Weight for BM25 in weighted-RRF fusion (hybrid mode)
            rrf_k: RRF constant (larger = flatter rank decay)
        """
        self.vectordb = vectordb
        self.parent_chunks = {p.chunk_id: p for p in parent_chunks}
        self.parent_to_children = parent_to_children
        self.top_k = top_k
        self.context_budget = context_budget
        self.retrieval_mode = retrieval_mode.lower()
        self.bm25_weight = max(0.0, min(1.0, bm25_weight))
        self.dense_weight = 1.0 - self.bm25_weight
        self.rrf_k = max(1, int(rrf_k))

        if self.retrieval_mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(
                f"Invalid retrieval_mode='{retrieval_mode}'. Choose from: dense, bm25, hybrid"
            )

        # Reverse index so a child can expand to multiple parents (cross-ref enrichment).
        self.child_to_parents = {}
        for parent_id, child_ids in (parent_to_children or {}).items():
            for child_id in child_ids or []:
                self.child_to_parents.setdefault(child_id, []).append(parent_id)

        # Child metadata cache + optional BM25 index.
        self.child_id_to_parent = {}
        self.child_id_to_text = {}
        self.child_ids = []
        self.bm25 = None

        # Metadata gives each child's host parent and the BM25 corpus. Fail loudly if it cannot
        # be read: silently switching a hybrid run to dense would change reported results.
        try:
            raw = self.vectordb.get(include=["metadatas", "documents"])
        except Exception as exc:
            raise RuntimeError(f"Could not read child metadata from the vector store: {exc}") from exc
        metadatas = raw.get("metadatas", []) if raw else []
        documents = raw.get("documents", []) if raw else []

        for metadata, text in zip(metadatas, documents):
            if not metadata:
                continue
            chunk_id = metadata.get("chunk_id")
            parent_id = metadata.get("parent_id")
            if not chunk_id:
                continue
            self.child_ids.append(chunk_id)
            self.child_id_to_parent[chunk_id] = parent_id
            self.child_id_to_text[chunk_id] = text or ""

        if self.retrieval_mode in {"bm25", "hybrid"} and self.child_ids:
            tokenized = [self._bm25_tokenize(self.child_id_to_text[cid]) for cid in self.child_ids]
            self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _bm25_tokenize(text: str) -> List[str]:
        """Tokenizer for mixed legal text (English + Japanese).

        - English/numeric tokens: regex word extraction
        - CJK spans: character bigrams for whitespace-free Japanese text
        """
        if not text:
            return []

        s = str(text).lower()
        tokens: List[str] = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", s)

        cjk_spans = re.findall(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+", s)
        for span in cjk_spans:
            if len(span) == 1:
                tokens.append(span)
            else:
                tokens.extend(span[i:i+2] for i in range(len(span) - 1))

        return tokens

    def _dense_child_ids(self, query: str) -> List[str]:
        docs = self.vectordb.similarity_search(query, k=self.top_k)
        seen = set()
        ranked = []
        for doc in docs:
            child_id = doc.metadata.get("chunk_id")
            if child_id and child_id not in seen:
                seen.add(child_id)
                ranked.append(child_id)
        return ranked[:self.top_k]

    def _bm25_child_ids(self, query: str) -> List[str]:
        if not self.bm25 or not self.child_ids:
            return []
        scores = self.bm25.get_scores(self._bm25_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        return [self.child_ids[idx] for idx in ranked_indices[:self.top_k]]

    def _rank_child_ids(self, query: str) -> List[str]:
        if self.retrieval_mode == "dense":
            return self._dense_child_ids(query)

        if self.retrieval_mode == "bm25":
            bm25_ids = self._bm25_child_ids(query)
            return bm25_ids if bm25_ids else self._dense_child_ids(query)

        dense_ids = self._dense_child_ids(query)
        bm25_ids = self._bm25_child_ids(query)

        if not bm25_ids:
            return dense_ids

        scores = {}
        for rank, child_id in enumerate(dense_ids, start=1):
            scores[child_id] = scores.get(child_id, 0.0) + self.dense_weight / (self.rrf_k + rank)
        for rank, child_id in enumerate(bm25_ids, start=1):
            scores[child_id] = scores.get(child_id, 0.0) + self.bm25_weight / (self.rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [child_id for child_id, _ in ranked[:self.top_k]]
    
    def get_relevant_documents(self, query: str):
        """
        Hierarchical retrieval pipeline with token budget constraint:
        
        1. Dense search on child chunks (sparse, high precision)
        2. Map children to parent chunks (deduplication)
        3. Return ONLY parent chunks as context (full sections)
        
        Args:
            query: User query
            
        Returns:
            List of parent documents only (children used only for ranking)
        """
        # Step 1: Rank child chunks (dense / bm25 / hybrid)
        ranked_child_ids = self._rank_child_ids(query)
        
        expanded_docs = []
        seen_parents = set()
        total_tokens = 0
        
        # Step 2: Map children to parents and add ONLY parents
        for child_id in ranked_child_ids:
            direct_parent_id = self.child_id_to_parent.get(child_id)

            # Pi(c) = {host(c)} U refs(c): the host parent first, then cross-referenced parents.
            candidate_parent_ids = [direct_parent_id] if direct_parent_id else []
            candidate_parent_ids.extend(
                pid for pid in (self.child_to_parents.get(child_id) or []) if pid != direct_parent_id
            )

            # Preserve rank order: expand parents in the order we encounter children.
            for parent_id in candidate_parent_ids:
                if not parent_id or parent_id in seen_parents:
                    continue

                parent_chunk = self.parent_chunks.get(parent_id)
                if not parent_chunk:
                    continue

                parent_tokens = parent_chunk.token_count

                # Only add parent if it doesn't exceed budget
                if total_tokens + parent_tokens <= self.context_budget:
                    from langchain_core.documents import Document
                    parent_doc = Document(
                        page_content=parent_chunk.text,
                        metadata={
                            'chunk_id': parent_chunk.chunk_id,
                            'chunk_type': 'parent',
                            'section_num': parent_chunk.section_num,
                            'section_title': parent_chunk.section_title,
                            'token_count': parent_chunk.token_count,
                            'contract_id': parent_chunk.contract_id,
                        }
                    )
                    expanded_docs.append(parent_doc)
                    total_tokens += parent_tokens
                    seen_parents.add(parent_id)
        
        return expanded_docs
    
    def invoke(self, query: str):
        """LangChain-compatible invoke method (alias for get_relevant_documents)"""
        return self.get_relevant_documents(query)


# End of module
