"""
Core summarization pipeline.

Pulls repo data from GitHub, sanitizes and chunks the content,
runs it through the LLM via Map-Reduce (or Iterative Refinement),
and assembles the final SummarizeResponse.

Both sync (JSON endpoint) and streaming (SSE) variants live here.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from app.core.config import SummarizationStrategy, get_settings
from app.core.logging import get_logger
from app.schemas.summarize import (
    RepoMetadata,
    SummarizationMeta,
    SummarizeRequest,
    SummarizeResponse,
)
from app.services.github_client import GitHubClient, parse_repo_url
from app.services.llm_factory import get_llm
from app.services.preprocessor import (
    build_file_tree_snippet,
    chunk_code_with_cast,
    chunk_text,
    estimate_tokens,
    is_code_file,
    sanitize_markdown,
)

logger = get_logger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert software architect and technical writer. "
    "Your task is to analyse GitHub repository documentation and produce "
    "clear, accurate, and insightful technical summaries for engineers "
    "and technical decision-makers. "
    "Be factual. Do not hallucinate library names or features. "
    "Write in concise, professional prose."
)

_MAP_PROMPT = (
    "Below is a section of documentation from a GitHub repository. "
    "Write a concise technical summary of what this section describes, "
    "focusing on purpose, architecture, and technologies mentioned.\n\n"
    "---\n{chunk}\n---"
)

_REDUCE_PROMPT = (
    "You have been given multiple partial summaries of different sections "
    "of a GitHub repository. Combine them into a single, coherent, "
    "well-structured technical summary. Include:\n"
    "1. Project purpose and core problem it solves\n"
    "2. Primary technologies and frameworks\n"
    "3. Architecture overview (if discernible)\n"
    "4. Key features and capabilities\n"
    "5. Development status and community health signals\n\n"
    "Partial summaries:\n---\n{combined}\n---\n\n"
    "Repository metadata:\n{metadata}"
)

_REFINE_INIT_PROMPT = (
    "Summarise the following repository documentation section:\n\n---\n{chunk}\n---"
)

_REFINE_STEP_PROMPT = (
    "Here is an existing summary of a repository:\n\n"
    "{existing_summary}\n\n"
    "Now refine and expand this summary using the following additional "
    "documentation section (add new details, correct any inconsistencies):\n\n"
    "---\n{chunk}\n---"
)

_EXTRACT_TECHNOLOGIES_PROMPT = (
    "From the following repository summary, extract a flat JSON array of "
    "the primary programming languages, frameworks, and tools mentioned. "
    "Return ONLY the JSON array, no explanation.\n\n{summary}"
)


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _llm_call(llm: BaseChatModel, user_prompt: str) -> str:
    """Single async LLM call returning stripped string output."""
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    result = await llm.ainvoke(messages)
    return StrOutputParser().invoke(result).strip()


def _format_metadata(meta: dict[str, Any]) -> str:
    lines = [
        f"Name: {meta.get('full_name', 'N/A')}",
        f"Description: {meta.get('description') or 'None'}",
        f"Stars: {meta.get('stargazers_count', 0):,}",
        f"Forks: {meta.get('forks_count', 0):,}",
        f"Primary Language: {meta.get('language') or 'N/A'}",
        f"Topics: {', '.join(meta.get('topics', []) or ['none'])}",
        f"License: {(meta.get('license') or {}).get('name', 'N/A')}",
        f"Archived: {meta.get('archived', False)}",
        f"Last Updated: {meta.get('updated_at', 'N/A')}",
    ]
    return "\n".join(lines)


async def _gather_content(
    client: GitHubClient,
    owner: str,
    repo: str,
    max_files: int,
) -> tuple[str, list[tuple[str, str]], list[dict], list[str]]:
    """
    Gather README + supplementary files from GitHub.

    Returns:
        doc_text:   Concatenated prose documentation (README, .md, .rst, .txt)
                    → chunked later with the recursive Markdown-aware splitter.
        code_files: List of (filename, content) tuples for source code files
                    → chunked later with ChunkHound's cAST algorithm.
        root_entries: Raw GitHub directory listing for file-tree rendering.
        warnings:   Non-fatal issues encountered during fetching.
    """
    warnings: list[str] = []

    # Parallel fetch: README + directory listing
    readme_task = asyncio.create_task(client.get_readme(owner, repo))
    dir_task = asyncio.create_task(client.get_directory_contents(owner, repo))
    readme, root_entries = await asyncio.gather(readme_task, dir_task)

    # --- README (always treated as prose documentation) ---
    doc_text = f"# README\n\n{readme}\n\n" if readme else ""
    if not readme:
        warnings.append("No README found; relying on metadata and file structure.")

    # --- Classify candidate files by chunking strategy ---
    # Prose docs  → Markdown-aware recursive splitter
    _DOC_EXTENSIONS = {".md", ".rst", ".txt"}
    # Config/manifests → prose splitter (structured but not AST-parseable prose)
    _CONFIG_EXTENSIONS = {".toml", ".yaml", ".yml"}
    # Priority filenames always fetched regardless of extension
    _PRIORITY_DOC_FILES = {
        "CONTRIBUTING.md",
        "ARCHITECTURE.md",
        "CHANGELOG.md",
        "go.mod",
        "Dockerfile",
    }
    _PRIORITY_CONFIG_FILES = {
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "docker-compose.yml",
        "docker-compose.yaml",
    }

    doc_candidates: list[dict] = []
    code_candidates: list[dict] = []

    for entry in root_entries:
        if entry.get("type") != "file":
            continue
        name: str = entry.get("name", "")
        if name == "README.md":
            continue  # already fetched above

        if (
            name in _PRIORITY_DOC_FILES
            or any(name.endswith(e) for e in _DOC_EXTENSIONS)
            or name in _PRIORITY_CONFIG_FILES
            or any(name.endswith(e) for e in _CONFIG_EXTENSIONS)
        ):
            doc_candidates.append(entry)
        elif is_code_file(name):
            code_candidates.append(entry)

    # Cap to max_files, prioritising docs first then code
    total_budget = max_files
    doc_candidates = doc_candidates[:total_budget]
    code_candidates = code_candidates[: max(0, total_budget - len(doc_candidates))]

    # Fetch all candidates in parallel
    all_candidates = doc_candidates + code_candidates
    if all_candidates:
        file_contents = await asyncio.gather(
            *[client.get_file_content(owner, repo, c["path"]) for c in all_candidates]
        )
    else:
        file_contents = []

    code_files: list[tuple[str, str]] = []

    for entry, content in zip(all_candidates, file_contents):
        if not content:
            continue
        name = entry.get("name", "")
        if entry in doc_candidates:
            doc_text += f"\n\n## {name}\n\n{content}"
        else:
            # Source code file → cAST chunking
            code_files.append((name, content))

    return doc_text, code_files, root_entries, warnings


# ── Main summarization strategies ─────────────────────────────────────────────


async def _map_reduce_summarize(
    llm: BaseChatModel,
    chunks: list[str],
    metadata_str: str,
) -> str:
    """
    Map phase: summarise each chunk independently (parallelised).
    Reduce phase: combine all mini-summaries into one final summary.
    """
    logger.info("map_reduce_start", num_chunks=len(chunks))

    # MAP — parallel LLM calls
    map_tasks = [_llm_call(llm, _MAP_PROMPT.format(chunk=chunk)) for chunk in chunks]
    partial_summaries: list[str] = await asyncio.gather(*map_tasks)

    # REDUCE
    combined = "\n\n---\n\n".join(f"Section {i + 1}:\n{s}" for i, s in enumerate(partial_summaries))
    final = await _llm_call(
        llm,
        _REDUCE_PROMPT.format(combined=combined, metadata=metadata_str),
    )
    logger.info("map_reduce_complete")
    return final


async def _refine_summarize(
    llm: BaseChatModel,
    chunks: list[str],
    metadata_str: str,
) -> str:
    """
    Iterative Refinement: build a running summary, refining it
    chunk by chunk for maximum contextual coherence.
    """
    logger.info("refine_start", num_chunks=len(chunks))

    summary = await _llm_call(llm, _REFINE_INIT_PROMPT.format(chunk=chunks[0]))

    for i, chunk in enumerate(chunks[1:], start=2):
        logger.debug("refine_step", step=i, total=len(chunks))
        summary = await _llm_call(
            llm,
            _REFINE_STEP_PROMPT.format(existing_summary=summary, chunk=chunk),
        )

    logger.info("refine_complete")
    return summary


async def _extract_technologies(llm: BaseChatModel, summary: str) -> list[str]:
    """Ask the LLM to return a JSON array of tech names from the summary."""
    import json

    raw = await _llm_call(
        llm,
        _EXTRACT_TECHNOLOGIES_PROMPT.format(summary=summary),
    )
    # Strip markdown code fences if present
    raw = raw.strip().strip("`").removeprefix("json").strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return [str(x) for x in result[:20]]
    except json.JSONDecodeError:
        pass
    return []


# ── Public entry point ────────────────────────────────────────────────────────


async def summarize_repo(
    request: SummarizeRequest, github_token: str | None = None
) -> SummarizeResponse:
    """
    Full summarization pipeline. Returns a structured SummarizeResponse.
    Called by the JSON (non-streaming) endpoint.
    """
    settings = get_settings()
    owner, repo_name = parse_repo_url(request.repo_url)
    max_files = request.max_files or settings.max_files_to_scan
    strategy = SummarizationStrategy(
        request.strategy.value if request.strategy else settings.summarization_strategy
    )

    llm = get_llm(
        provider=request.llm_provider.value if request.llm_provider else None,
        model=request.llm_model,
    )

    client = GitHubClient(github_token=github_token)

    logger.info("summarize_start", owner=owner, repo=repo_name, strategy=strategy)

    # ── 1. Fetch data (parallel: metadata + content + languages) ─────────────
    (
        (doc_text, code_files, root_entries, warnings),
        raw_meta,
        languages,
    ) = await asyncio.gather(
        _gather_content(client, owner, repo_name, max_files),
        client.get_repo_metadata(owner, repo_name),
        client.get_languages(owner, repo_name),
    )

    # ── 2. Pre-process & chunk ────────────────────────────────────────────────
    #
    # Strategy:
    #   • Prose documentation (README, .md, .rst, .txt, configs)
    #     → sanitise Markdown → recursive heading-aware splitter
    #   • Source code files (.py, .js, .go, .rs, …)
    #     → ChunkHound cAST algorithm (Tree-sitter, AST-aware)
    #     → each chunk is annotated with file name + symbol name
    #
    chunk_size = settings.chunk_size_tokens

    # --- Doc chunks ---
    clean_doc = sanitize_markdown(doc_text)
    doc_chunks = chunk_text(clean_doc, chunk_size=chunk_size) if clean_doc.strip() else []

    # --- Code chunks (cAST) ---
    code_chunks: list[str] = []
    for fname, code_content in code_files:
        code_chunks.extend(chunk_code_with_cast(code_content, fname, max_tokens=chunk_size))

    # Combine: docs first (high-level context), then code (implementation detail)
    chunks = doc_chunks + code_chunks

    # Guard against token budget overrun across all chunks
    total_tokens = sum(estimate_tokens(c) for c in chunks)
    if total_tokens > settings.max_total_tokens:
        warnings.append(
            f"Content truncated: {total_tokens:,} estimated tokens exceeded "
            f"limit of {settings.max_total_tokens:,}. Dropping tail chunks."
        )
        budget = 0
        kept: list[str] = []
        for c in chunks:
            t = estimate_tokens(c)
            if budget + t > settings.max_total_tokens:
                break
            kept.append(c)
            budget += t
        chunks = kept
        total_tokens = budget

    if not chunks:
        # Ultimate fallback: metadata-only summary
        chunks = [f"Repository metadata:\n{_format_metadata(raw_meta)}"]
        warnings.append("No content found; summary is based on metadata only.")

    metadata_str = _format_metadata(raw_meta)

    # ── 3. Summarise ──────────────────────────────────────────────────────
    if strategy == SummarizationStrategy.MAP_REDUCE:
        summary = await _map_reduce_summarize(llm, chunks, metadata_str)
    else:
        summary = await _refine_summarize(llm, chunks, metadata_str)

    technologies = await _extract_technologies(llm, summary)

    # ── 4. Build response ─────────────────────────────────────────────────
    license_info = raw_meta.get("license") or {}
    meta = RepoMetadata(
        owner=owner,
        name=repo_name,
        full_name=raw_meta.get("full_name", f"{owner}/{repo_name}"),
        description=raw_meta.get("description"),
        stars=raw_meta.get("stargazers_count", 0),
        forks=raw_meta.get("forks_count", 0),
        open_issues=raw_meta.get("open_issues_count", 0),
        topics=raw_meta.get("topics", []),
        default_branch=raw_meta.get("default_branch", "main"),
        language=raw_meta.get("language"),
        languages=languages,
        is_archived=raw_meta.get("archived", False),
        license_name=license_info.get("name"),
        homepage=raw_meta.get("homepage"),
        created_at=raw_meta.get("created_at"),
        updated_at=raw_meta.get("updated_at"),
    )

    llm_provider_name = (
        request.llm_provider.value if request.llm_provider else settings.default_llm_provider.value
    )
    llm_model_name = request.llm_model or settings.default_llm_model

    summ_meta = SummarizationMeta(
        strategy_used=strategy.value,
        llm_provider=llm_provider_name,
        llm_model=llm_model_name,
        total_input_tokens_estimated=total_tokens,
        chunks_processed=len(chunks),
        files_scanned=len(root_entries),
        cache_hit=False,
    )

    # Log code-vs-doc chunk split for observability
    logger.info(
        "summarize_complete",
        owner=owner,
        repo=repo_name,
        tokens=total_tokens,
        doc_chunks=len(doc_chunks),
        code_chunks=len(code_chunks),
        total_chunks=len(chunks),
        code_files_cast=len(code_files),
    )

    file_tree = None
    if request.include_file_tree and root_entries:
        file_tree = build_file_tree_snippet(root_entries)

    return SummarizeResponse(
        repo_url=request.repo_url,
        metadata=meta,
        summary=summary,
        key_technologies=technologies,
        file_tree_snippet=file_tree,
        summarization_meta=summ_meta,
        warnings=warnings,
    )


# ── Streaming variant ─────────────────────────────────────────────────────────


async def summarize_repo_stream(
    request: SummarizeRequest, github_token: str | None = None
) -> AsyncIterator[str]:
    """
    Streaming summarization using Server-Sent Events.
    Yields JSON-serialisable event strings.

    Events:
      {"event": "status",  "data": "..."}   — progress updates
      {"event": "chunk",   "data": "..."}   — summary token chunks
      {"event": "done",    "data": {...}}    — final metadata object
      {"event": "error",   "data": "..."}   — on failure
    """
    import json

    settings = get_settings()
    owner, repo_name = parse_repo_url(request.repo_url)
    max_files = request.max_files or settings.max_files_to_scan
    strategy = SummarizationStrategy(
        request.strategy.value if request.strategy else settings.summarization_strategy
    )

    llm = get_llm(
        provider=request.llm_provider.value if request.llm_provider else None,
        model=request.llm_model,
    )
    client = GitHubClient(github_token=github_token)

    def _event(event_type: str, data: Any) -> str:
        return json.dumps({"event": event_type, "data": data})

    try:
        yield _event("status", f"Fetching repository data for {owner}/{repo_name}…")

        (
            (doc_text, code_files, root_entries, warnings),
            raw_meta,
            languages,
        ) = await asyncio.gather(
            _gather_content(client, owner, repo_name, max_files),
            client.get_repo_metadata(owner, repo_name),
            client.get_languages(owner, repo_name),
        )

        yield _event("status", "Sanitising and chunking content…")
        chunk_size = settings.chunk_size_tokens

        # Prose docs — Markdown-aware recursive splitter
        clean_doc = sanitize_markdown(doc_text)
        doc_chunks = chunk_text(clean_doc, chunk_size=chunk_size) if clean_doc.strip() else []

        # Code files — ChunkHound cAST (AST-aware, per-file)
        code_chunks: list[str] = []
        for fname, code_content in code_files:
            code_chunks.extend(chunk_code_with_cast(code_content, fname, max_tokens=chunk_size))

        chunks = doc_chunks + code_chunks

        # Token budget guard
        total_tokens = sum(estimate_tokens(c) for c in chunks)
        if total_tokens > settings.max_total_tokens:
            budget = 0
            kept: list[str] = []
            for c in chunks:
                t = estimate_tokens(c)
                if budget + t > settings.max_total_tokens:
                    break
                kept.append(c)
                budget += t
            chunks = kept
            total_tokens = budget
            warnings.append("Content was truncated due to token limit.")

        if not chunks:
            chunks = [f"Repository metadata:\n{_format_metadata(raw_meta)}"]

        yield _event(
            "status",
            f"Summarising {len(chunks)} chunk(s) using {strategy.value} strategy…",
        )

        metadata_str = _format_metadata(raw_meta)

        # Stream the reduce/final step token by token
        if strategy == SummarizationStrategy.MAP_REDUCE:
            # Map phase (parallel, not streamed)
            map_tasks = [_llm_call(llm, _MAP_PROMPT.format(chunk=c)) for c in chunks]
            partial_summaries = await asyncio.gather(*map_tasks)
            combined = "\n\n---\n\n".join(
                f"Section {i + 1}:\n{s}" for i, s in enumerate(partial_summaries)
            )
            reduce_prompt = _REDUCE_PROMPT.format(combined=combined, metadata=metadata_str)
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=reduce_prompt),
            ]
            summary_tokens: list[str] = []
            async for token in llm.astream(messages):
                text_chunk = token.content if hasattr(token, "content") else str(token)
                if text_chunk:
                    summary_tokens.append(text_chunk)
                    yield _event("chunk", text_chunk)
            summary = "".join(summary_tokens)

        else:  # REFINE
            summary = await _llm_call(llm, _REFINE_INIT_PROMPT.format(chunk=chunks[0]))
            yield _event("chunk", summary)

            for i, chunk in enumerate(chunks[1:], start=2):
                yield _event("status", f"Refining with section {i}/{len(chunks)}…")
                refine_prompt = _REFINE_STEP_PROMPT.format(existing_summary=summary, chunk=chunk)
                messages = [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=refine_prompt),
                ]
                new_tokens: list[str] = []
                async for token in llm.astream(messages):
                    text_chunk = token.content if hasattr(token, "content") else str(token)
                    if text_chunk:
                        new_tokens.append(text_chunk)
                        yield _event("chunk", text_chunk)
                summary = "".join(new_tokens)

        technologies = await _extract_technologies(llm, summary)

        license_info = raw_meta.get("license") or {}
        done_payload = {
            "key_technologies": technologies,
            "metadata": {
                "owner": owner,
                "name": repo_name,
                "full_name": raw_meta.get("full_name"),
                "stars": raw_meta.get("stargazers_count", 0),
                "language": raw_meta.get("language"),
                "languages": languages,
                "topics": raw_meta.get("topics", []),
                "license_name": license_info.get("name"),
            },
            "summarization_meta": {
                "strategy_used": strategy.value,
                "llm_provider": (
                    request.llm_provider.value
                    if request.llm_provider
                    else settings.default_llm_provider.value
                ),
                "llm_model": request.llm_model or settings.default_llm_model,
                "total_input_tokens_estimated": total_tokens,
                "chunks_processed": len(chunks),
                "files_scanned": len(root_entries),
            },
            "warnings": warnings,
        }
        yield _event("done", done_payload)

    except Exception as exc:
        logger.error("summarize_stream_error", error=str(exc), exc_info=True)
        yield _event("error", str(exc))
