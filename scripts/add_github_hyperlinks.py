#!/usr/bin/env python3
"""
Script to add GitHub hyperlinks to file references in documentation.

This script processes markdown files and converts file references to GitHub hyperlinks.
It handles several patterns:
- Plain filenames: `00_download_datasets.py` → [`00_download_datasets.py`](https://github.com/...)
- File paths: `scripts/convert.py` → [`scripts/convert.py`](https://github.com/...)
- File paths with line numbers: `misc/config.py`, lines 171-176 → hyperlink with line range
- File paths with function references: `misc/config.py::Config.__init__` → hyperlink
"""

import re
import os
from pathlib import Path
from typing import Tuple, Optional

# GitHub repository base URL
GITHUB_REPO = "https://github.com/jgonsior/olympic-games-of-active-learning"
GITHUB_BRANCH = "main"  # Use main branch for stable links

def get_github_url(file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """
    Generate GitHub URL for a file path with optional line numbers.
    
    Args:
        file_path: Path to the file (e.g., "misc/config.py")
        start_line: Starting line number (optional)
        end_line: Ending line number (optional)
        
    Returns:
        Full GitHub URL with line anchors if provided
    """
    base_url = f"{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{file_path}"
    
    if start_line is not None:
        if end_line is not None and end_line != start_line:
            base_url += f"#L{start_line}-L{end_line}"
        else:
            base_url += f"#L{start_line}"
    
    return base_url

def process_source_citation(match: re.Match) -> str:
    """
    Process source citations like:
    (source: `misc/config.py::Config.__init__`, lines 171-176)
    or
    (source: `misc/config.py`, lines 171-176)
    or
    (source: `misc/config.py::Config`)
    
    Returns the replacement text with hyperlink.
    """
    prefix = match.group(1)  # "(source: "
    file_path = match.group(2)  # "misc/config.py"
    class_func = match.group(3)  # "::Config.__init__" or None
    lines_text = match.group(4)  # ", lines 171-176" or None
    suffix = match.group(5)  # ")" or "; ..." 
    
    # Parse line numbers if present
    start_line = None
    end_line = None
    if lines_text:
        lines_match = re.search(r'(\d+)(?:-(\d+))?', lines_text)
        if lines_match:
            start_line = int(lines_match.group(1))
            if lines_match.group(2):
                end_line = int(lines_match.group(2))
    
    # Create display text
    display_text = file_path
    if class_func:
        display_text += class_func
    
    # Generate GitHub URL
    github_url = get_github_url(file_path, start_line, end_line)
    
    # Build replacement text
    replacement = f"{prefix}[`{display_text}`]({github_url})"
    if lines_text:
        replacement += lines_text
    replacement += suffix
    
    return replacement

def process_inline_file_reference(match: re.Match, base_dir: Path) -> str:
    """
    Process inline file references like:
    `misc/config.py` or `00_download_datasets.py`
    
    Only convert if the file actually exists in the repository.
    
    Returns the replacement text with hyperlink.
    """
    file_path = match.group(1)
    
    # Check if this looks like a file path (contains .py, .yaml, .cfg, .md, etc. or ends with /)
    is_file = (
        file_path.endswith('.py') or
        file_path.endswith('.yaml') or
        file_path.endswith('.yml') or
        file_path.endswith('.cfg') or
        file_path.endswith('.md') or
        file_path.endswith('.txt') or
        file_path.endswith('.toml') or
        file_path.endswith('.json') or
        file_path.endswith('.sh') or
        file_path.endswith('/') or
        '/' in file_path
    )
    
    if not is_file:
        return match.group(0)  # Return unchanged
    
    # Check if file exists in repository
    full_path = base_dir / file_path.rstrip('/')
    if not full_path.exists():
        return match.group(0)  # Return unchanged if file doesn't exist
    
    # Generate GitHub URL
    github_url = get_github_url(file_path.rstrip('/'))
    
    return f"[`{file_path}`]({github_url})"

def process_markdown_file(file_path: Path, repo_root: Path, dry_run: bool = False) -> Tuple[int, int]:
    """
    Process a markdown file and add GitHub hyperlinks to file references.
    
    Args:
        file_path: Path to the markdown file
        repo_root: Path to the repository root
        dry_run: If True, only count changes without modifying files
        
    Returns:
        Tuple of (source_citations_converted, inline_refs_converted)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    source_citations = 0
    inline_refs = 0
    
    # Extract code blocks (to avoid converting references inside code)
    code_blocks = []
    code_block_pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    
    def code_block_replacer(m):
        code_blocks.append(m.group(0))
        return f'CODE_BLOCK_PLACEHOLDER_{len(code_blocks)-1}'
    
    content = code_block_pattern.sub(code_block_replacer, content)
    
    # Pattern 1: Source citations with optional class/function and line numbers
    # (source: `misc/config.py::Config.__init__`, lines 171-176)
    # (source: `misc/config.py`, lines 171-176)
    # (source: `misc/config.py::Config`)
    # Also handle multiple references in one source line
    source_pattern = re.compile(
        r'(\(source:\s+)`([^`]+?)(::[\w.]+)?`(,?\s*lines?\s+\d+(?:-\d+)?)?([);,])',
        re.IGNORECASE
    )
    
    def source_replacer(m):
        nonlocal source_citations
        # Skip if already a hyperlink
        if '[`' in m.group(0):
            return m.group(0)
        source_citations += 1
        return process_source_citation(m)
    
    # Keep applying the pattern until no more matches (to handle multiple refs per line)
    prev_content = None
    while prev_content != content:
        prev_content = content
        content = source_pattern.sub(source_replacer, content)
    
    # Pattern 2: Inline file references (only if not already hyperlinked)
    # `misc/config.py` or `00_download_datasets.py`
    # But NOT inside hyperlinks or source citations
    inline_pattern = re.compile(r'(?<!\[)`([^`\n]+?)`(?!\])')
    
    def inline_replacer(m):
        nonlocal inline_refs
        # Skip if this is part of a source citation we already processed
        # or if already a hyperlink
        full_match = m.group(0)
        
        # Check if we're inside a markdown link
        start_pos = m.start()
        # Look back to see if we're in a link
        lookback = content[max(0, start_pos-100):start_pos]
        if '(' in lookback and lookback.rfind('(') > lookback.rfind(')'):
            return full_match
        
        result = process_inline_file_reference(m, repo_root)
        if result != full_match:
            inline_refs += 1
        return result
    
    content = inline_pattern.sub(inline_replacer, content)
    
    # Restore code blocks
    for i, block in enumerate(code_blocks):
        content = content.replace(f'CODE_BLOCK_PLACEHOLDER_{i}', block)
    
    # Write back if changed and not dry run
    if content != original_content and not dry_run:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    return source_citations, inline_refs

def main():
    """Main function to process all documentation files."""
    # Get repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    # Process all markdown files in docs/ and README.md
    docs_dir = repo_root / 'docs'
    readme_file = repo_root / 'README.md'
    
    total_source_citations = 0
    total_inline_refs = 0
    files_processed = 0
    
    # Process all markdown files in docs/
    for md_file in docs_dir.rglob('*.md'):
        source_count, inline_count = process_markdown_file(md_file, repo_root)
        if source_count > 0 or inline_count > 0:
            print(f"✓ {md_file.relative_to(repo_root)}: {source_count} source citations, {inline_count} inline refs")
            total_source_citations += source_count
            total_inline_refs += inline_count
            files_processed += 1
    
    # Process README.md
    if readme_file.exists():
        source_count, inline_count = process_markdown_file(readme_file, repo_root)
        if source_count > 0 or inline_count > 0:
            print(f"✓ {readme_file.relative_to(repo_root)}: {source_count} source citations, {inline_count} inline refs")
            total_source_citations += source_count
            total_inline_refs += inline_count
            files_processed += 1
    
    print("\n" + "="*70)
    print(f"✓ Processed {files_processed} files")
    print(f"✓ Converted {total_source_citations} source citations")
    print(f"✓ Converted {total_inline_refs} inline file references")
    print(f"✓ Total: {total_source_citations + total_inline_refs} hyperlinks added")

if __name__ == '__main__':
    main()
