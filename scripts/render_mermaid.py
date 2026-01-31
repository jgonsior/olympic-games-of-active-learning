#!/usr/bin/env python3
"""
Pre-render Mermaid diagrams to SVG files using mermaid-cli (mmdc).

This script:
1. Scans markdown files for mermaid code blocks
2. Extracts diagram code to temporary files
3. Uses mmdc (mermaid-cli) to render SVG files
4. Saves SVG files to docs/images/mermaid/ directory

The static SVG files serve as fallbacks if the CDN is unavailable
or JavaScript is disabled.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
import hashlib


def find_mermaid_blocks(markdown_content):
    """
    Find all mermaid code blocks in markdown content.
    
    Returns: list of tuples (mermaid_code, start_pos, end_pos)
    """
    # Pattern to match ```mermaid ... ``` blocks
    pattern = r'```mermaid\s*\n(.*?)```'
    matches = []
    
    for match in re.finditer(pattern, markdown_content, re.DOTALL):
        mermaid_code = match.group(1).strip()
        matches.append((mermaid_code, match.start(), match.end()))
    
    return matches


def generate_svg_filename(mermaid_code, markdown_file):
    """
    Generate a unique but reproducible filename for the SVG.
    
    Uses hash of content + source file for uniqueness.
    """
    # Create a hash of the mermaid code for uniqueness
    code_hash = hashlib.md5(mermaid_code.encode()).hexdigest()[:8]
    
    # Use markdown filename as prefix
    md_name = Path(markdown_file).stem
    
    return f"{md_name}-{code_hash}.svg"


def render_mermaid_to_svg(mermaid_code, output_file, puppeteer_config=None):
    """
    Render mermaid code to SVG using mmdc CLI.
    
    Args:
        mermaid_code: The mermaid diagram code
        output_file: Path to output SVG file
        puppeteer_config: Path to puppeteer config file (optional)
    
    Returns:
        True if successful, False otherwise
    """
    # Create a temporary input file
    temp_dir = Path("/tmp/mermaid_render")
    temp_dir.mkdir(exist_ok=True)
    
    temp_input = temp_dir / "diagram.mmd"
    temp_input.write_text(mermaid_code)
    
    try:
        # Run mmdc to render the diagram
        # -i: input file, -o: output file, -t: theme, -b: background color
        # -p: puppeteer config for --no-sandbox in CI environments
        cmd = [
            "mmdc",
            "-i", str(temp_input),
            "-o", str(output_file),
            "-t", "default",
            "-b", "transparent"
        ]
        
        if puppeteer_config and puppeteer_config.exists():
            cmd.extend(["-p", str(puppeteer_config)])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"  ✓ Rendered: {output_file.name}")
            return True
        else:
            print(f"  ✗ Error rendering {output_file.name}:")
            print(f"    {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout rendering {output_file.name}")
        return False
    except FileNotFoundError:
        print("  ✗ Error: mmdc command not found. Install with: npm install -g @mermaid-js/mermaid-cli")
        return False
    finally:
        # Clean up temp file
        if temp_input.exists():
            temp_input.unlink()


def process_markdown_file(markdown_file, output_dir, puppeteer_config=None):
    """
    Process a single markdown file to extract and render mermaid diagrams.
    
    Args:
        markdown_file: Path to the markdown file
        output_dir: Directory to save SVG files
        puppeteer_config: Path to puppeteer config file (optional)
    
    Returns:
        Number of diagrams rendered
    """
    content = Path(markdown_file).read_text()
    mermaid_blocks = find_mermaid_blocks(content)
    
    if not mermaid_blocks:
        return 0
    
    print(f"\nProcessing: {markdown_file}")
    print(f"  Found {len(mermaid_blocks)} mermaid diagram(s)")
    
    rendered_count = 0
    for i, (mermaid_code, _, _) in enumerate(mermaid_blocks, 1):
        svg_filename = generate_svg_filename(mermaid_code, markdown_file)
        svg_path = output_dir / svg_filename
        
        if render_mermaid_to_svg(mermaid_code, svg_path, puppeteer_config):
            rendered_count += 1
    
    return rendered_count


def main():
    """Main function to process all markdown files."""
    # Setup paths
    repo_root = Path(__file__).parent.parent
    docs_dir = repo_root / "docs"
    output_dir = docs_dir / "images" / "mermaid"
    puppeteer_config = repo_root / "puppeteer-config.json"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Pre-rendering Mermaid Diagrams to SVG")
    print("=" * 70)
    
    # Find all markdown files with mermaid diagrams (recursively)
    markdown_files = []
    for md_file in docs_dir.glob("**/*.md"):
        content = md_file.read_text()
        if "```mermaid" in content:
            markdown_files.append(md_file)
    
    if not markdown_files:
        print("\nNo markdown files with mermaid diagrams found.")
        return 0
    
    # Process each file
    total_rendered = 0
    for md_file in markdown_files:
        total_rendered += process_markdown_file(md_file, output_dir, puppeteer_config)
    
    print("\n" + "=" * 70)
    print(f"Complete! Rendered {total_rendered} diagram(s)")
    print(f"SVG files saved to: {output_dir}")
    print("=" * 70)
    
    return 0 if total_rendered > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
