"""
Text pre-processing pipeline for repository content.

Strips markdown badges, social pixels, tracking links, and HTML noise.
Then chunks the cleaned text — prose docs go through a recursive
markdown-aware splitter; source code uses ChunkHound's cAST algorithm
(tree-sitter based) to split on function/class boundaries.

Also handles token counting (tiktoken) and file-tree rendering.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import tiktoken

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Compiled regex patterns ───────────────────────────────────────────────────

# Badge patterns (Markdown image syntax pointing to shields.io / badges)
_BADGE_MD = re.compile(
    r"!\[.*?\]\(https?://(img\.shields\.io|shields\.io|badge\.fury\.io"
    r"|travis-ci\.(org|com)|circleci\.com|codecov\.io"
    r"|github\.com/.+/(workflows|actions))[^\)]*\)",
    re.IGNORECASE,
)
# HTML <img> badge tags
_BADGE_HTML = re.compile(
    r"<img[^>]+(?:shields\.io|badge|build-status|travis)[^>]*>",
    re.IGNORECASE,
)
# Bare image links that are decorative (social icons, etc.)
_ICON_MD = re.compile(
    r"!\[(?:twitter|linkedin|discord|slack|npm|pypi|github)[^\]]*\]\([^\)]+\)",
    re.IGNORECASE,
)
# HTML comments
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# Script / style blocks
_SCRIPT_STYLE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
# Bare HTML tags (keep text content)
_HTML_TAG = re.compile(r"<[^>]+>")
# Multiple blank lines → single blank line
_MULTI_BLANK = re.compile(r"\n{3,}")
# Tracking pixel patterns (1x1 images)
_TRACKING_PIXEL = re.compile(
    r"!\[[^\]]*\]\(https?://[^\)]+[?&](?:utm_|pixel|track)[^\)]*\)",
    re.IGNORECASE,
)


def sanitize_markdown(text: str) -> str:
    """
    Clean raw Markdown / HTML documentation for LLM consumption.
    Returns cleaned string.
    """
    if not text:
        return ""

    # Remove HTML comment blocks first (may contain badges)
    text = _HTML_COMMENT.sub("", text)
    text = _SCRIPT_STYLE.sub("", text)

    # Remove badges and decorative images
    text = _BADGE_MD.sub("", text)
    text = _BADGE_HTML.sub("", text)
    text = _ICON_MD.sub("", text)
    text = _TRACKING_PIXEL.sub("", text)

    # Strip remaining HTML tags
    text = _HTML_TAG.sub("", text)

    # Collapse excessive whitespace
    text = _MULTI_BLANK.sub("\n\n", text)
    text = text.strip()

    return text


# ── Token estimation ──────────────────────────────────────────────────────────

_ENCODER: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        try:
            _ENCODER = tiktoken.encoding_for_model("gpt-4o")
        except KeyError:
            _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def estimate_tokens(text: str) -> int:
    """Return an approximate token count for the given text."""
    try:
        return len(_get_encoder().encode(text))
    except Exception:
        # Fallback: 1 token ≈ 4 chars
        return max(1, len(text) // 4)


# ── Recursive chunking ────────────────────────────────────────────────────────

_SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]


def _split_on_separators(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """
    Recursively split text at the first separator that produces chunks
    small enough, falling back to finer separators as needed.
    Mirrors LangChain's RecursiveCharacterTextSplitter logic.
    """
    if not separators:
        # No separator left — hard split by character count
        enc = _get_encoder()
        tokens = enc.encode(text)
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunks.append(enc.decode(tokens[i : i + chunk_size]))
        return chunks

    sep = separators[0]
    rest = separators[1:]

    parts = text.split(sep) if sep else list(text)
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part).lstrip(sep) if current else part
        if estimate_tokens(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Part itself may be too big → recurse
            if estimate_tokens(part) > chunk_size:
                chunks.extend(_split_on_separators(part, rest, chunk_size))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """
    Split text into overlapping chunks sized ≤ chunk_size tokens.
    Uses recursive splitting on Markdown heading boundaries first.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size_tokens
    overlap = overlap if overlap is not None else settings.chunk_overlap_tokens

    raw_chunks = _split_on_separators(text, _SEPARATORS, chunk_size)

    if overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    # Apply overlap: prepend tail of previous chunk to next chunk
    result: list[str] = [raw_chunks[0]]
    enc = _get_encoder()

    for i in range(1, len(raw_chunks)):
        prev_tokens = enc.encode(raw_chunks[i - 1])
        overlap_text = (
            enc.decode(prev_tokens[-overlap:]) if len(prev_tokens) > overlap else raw_chunks[i - 1]
        )
        result.append(overlap_text + "\n\n" + raw_chunks[i])

    return result


# ── File-tree builder ─────────────────────────────────────────────────────────


def build_file_tree_snippet(
    entries: list[dict],
    max_depth: int = 2,
    max_entries: int = 40,
) -> str:
    """
    Render a condensed ASCII file tree from GitHub contents API entries.
    Excludes common noise directories (node_modules, .git, __pycache__, etc.).
    """
    _IGNORED = {
        "node_modules",
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".venv",
        "venv",
    }
    lines: list[str] = []

    def _render(items: list[dict], prefix: str, depth: int) -> None:
        if depth > max_depth or len(lines) >= max_entries:
            return
        for item in items[:max_entries]:
            name = item.get("name", "")
            if name in _IGNORED:
                continue
            kind = "📁" if item.get("type") == "dir" else "📄"
            lines.append(f"{prefix}{kind} {name}")

    _render(entries, "", 0)
    return "\n".join(lines) or "(no file tree available)"


# ── cAST Code Chunker (ChunkHound) ────────────────────────────────────────────
#
# ChunkHound's UniversalParser implements the cAST algorithm from CMU / Augment
# Code research.  It uses Tree-sitter to parse source code into an AST, then
# segments it at meaningful semantic boundaries (functions, classes, modules)
# rather than at arbitrary character offsets.
#
# Integration strategy:
#   • ChunkHound operates on files on disk, not in-memory strings.
#   • We write fetched code content to a NamedTemporaryFile with the correct
#     extension, parse it, collect the chunk texts, then delete the temp file.
#   • Each chunk is annotated with a context header so the LLM knows the file
#     name and the symbol being described.
#   • On any import error or runtime failure we silently fall back to the
#     existing recursive text splitter — the API never breaks due to ChunkHound.
#
# Supported extensions (ChunkHound 4.x / 29 languages):
#   Code:   .py .js .jsx .ts .tsx .java .kt .groovy .c .cpp .cs .go .rs
#           .hs .swift .sh .m .php .vue .svelte .zig .matlab .mk
#   Config: .json .yaml .yml .toml .hcl
#   Docs:   .md .rst .txt  (these go through the Markdown-aware splitter above)

# Extensions that ChunkHound's cAST parser handles well.
# Markdown / rst / txt are intentionally excluded — the recursive splitter
# respects Markdown heading boundaries better for prose documents.
CAST_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        # ── Compiled languages ──────────────────────────────────────
        ".py",
        ".pyx",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".java",
        ".kt",
        ".groovy",
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".cxx",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".hs",
        ".swift",
        ".sh",
        ".bash",
        ".php",
        ".vue",
        ".svelte",
        ".zig",
        # ── Config / data (AST-parseable) ────────────────────────────
        ".toml",
        ".json",
        ".yaml",
        ".yml",
        ".hcl",
    }
)


def is_code_file(filename: str) -> bool:
    """Return True if this filename should be chunked with the cAST chunker."""
    return Path(filename).suffix.lower() in CAST_SUPPORTED_EXTENSIONS


# ── Lazy singleton for the ChunkHound parser ─────────────────────────────────

_CAST_PARSER: object | None = None  # type: ignore[type-arg]
_CAST_AVAILABLE: bool | None = None  # None = not yet checked


def _get_cast_parser() -> object | None:
    """
    Return a cached UniversalParser instance, or None if ChunkHound is not
    installed / importable.  Logs a one-time warning on first failure.
    """
    global _CAST_PARSER, _CAST_AVAILABLE

    if _CAST_AVAILABLE is True:
        return _CAST_PARSER

    if _CAST_AVAILABLE is False:
        return None

    try:
        from chunkhound.parsing.universal_parser import UniversalParser  # type: ignore

        _CAST_PARSER = UniversalParser()
        _CAST_AVAILABLE = True
        logger.info("cast_chunker_ready", backend="chunkhound")
        return _CAST_PARSER
    except ImportError:
        _CAST_AVAILABLE = False
        logger.warning(
            "chunkhound_not_installed",
            msg="Falling back to recursive text splitter for code files. "
            "Install with: pip install chunkhound>=4.0.0",
        )
        return None
    except Exception as exc:
        _CAST_AVAILABLE = False
        logger.warning("cast_parser_init_failed", error=str(exc))
        return None


# ── Public cAST chunking function ─────────────────────────────────────────────


def chunk_code_with_cast(
    content: str,
    filename: str,
    max_tokens: int | None = None,
) -> list[str]:
    """
    Chunk source code using ChunkHound's cAST algorithm.

    Writes ``content`` to a temporary file with the correct extension,
    invokes ChunkHound's ``UniversalParser.parse_file()``, and returns a
    list of text chunks — each prefixed with a context header so the LLM
    knows the file and symbol name.

    Falls back to ``chunk_text()`` on any error (import, parse, I/O).

    Args:
        content:   Raw source code string as fetched from the GitHub API.
        filename:  Original filename (e.g. "main.py").  Used for:
                   - selecting the correct Tree-sitter grammar via extension
                   - the context header prepended to each chunk
        max_tokens: Maximum tokens per chunk.  Defaults to settings value.
                    Chunks that exceed this are further split with chunk_text().

    Returns:
        List of non-empty chunk strings, ready for LLM consumption.
    """
    settings = get_settings()
    max_tokens = max_tokens or settings.chunk_size_tokens

    parser = _get_cast_parser()
    if parser is None or not content.strip():
        # ChunkHound unavailable — fall back gracefully
        return chunk_text(content, chunk_size=max_tokens)

    suffix = Path(filename).suffix or ".txt"
    raw_chunks: list[str] = []

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            encoding="utf-8",
            delete=False,
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            # UniversalParser.parse_file() returns a list of CodeChunk objects.
            # Each CodeChunk has at minimum:
            #   .content    str   — the raw source text of the chunk
            #   .name       str   — symbol name (function/class/module name)
            #   .chunk_type str   — e.g. "function", "class", "module", "block"
            #   .start_line int   — 1-based line number
            #   .end_line   int   — 1-based line number (inclusive)
            cast_chunks = parser.parse_file(tmp_path)  # type: ignore[union-attr]
        finally:
            tmp_path.unlink(missing_ok=True)

        if not cast_chunks:
            logger.debug("cast_no_chunks", filename=filename, fallback=True)
            return chunk_text(content, chunk_size=max_tokens)

        for chunk in cast_chunks:
            chunk_content: str = getattr(chunk, "content", "") or ""
            chunk_name: str = getattr(chunk, "name", "") or ""
            chunk_type: str = getattr(chunk, "chunk_type", "") or ""
            start_line: int = getattr(chunk, "start_line", 0) or 0
            end_line: int = getattr(chunk, "end_line", 0) or 0

            if not chunk_content.strip():
                continue

            # Build a rich context header for the LLM
            header_parts = [f"# File: {filename}"]
            if chunk_name:
                label = f"{chunk_type}: {chunk_name}" if chunk_type else chunk_name
                header_parts.append(f"# Symbol: {label}")
            if start_line:
                header_parts.append(f"# Lines: {start_line}–{end_line}")
            header = "\n".join(header_parts)

            annotated = f"{header}\n\n{chunk_content}"

            # If a single cAST chunk exceeds max_tokens, split it further
            if estimate_tokens(annotated) > max_tokens:
                sub_chunks = chunk_text(chunk_content, chunk_size=max_tokens)
                for i, sub in enumerate(sub_chunks, 1):
                    raw_chunks.append(f"{header} (part {i}/{len(sub_chunks)})\n\n{sub}")
            else:
                raw_chunks.append(annotated)

        logger.info(
            "cast_chunked",
            filename=filename,
            cast_chunks=len(cast_chunks),
            output_chunks=len(raw_chunks),
        )
        return raw_chunks if raw_chunks else chunk_text(content, chunk_size=max_tokens)

    except Exception as exc:
        logger.warning(
            "cast_chunk_failed",
            filename=filename,
            error=str(exc),
            fallback="recursive_splitter",
        )
        return chunk_text(content, chunk_size=max_tokens)
