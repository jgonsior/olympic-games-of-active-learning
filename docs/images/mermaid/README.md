# Pre-rendered Mermaid Diagrams

This directory contains static SVG files pre-rendered from Mermaid diagrams in the documentation.

## Purpose

These SVG files serve as static fallbacks for the Mermaid diagrams, providing:

- **CDN-independent rendering**: Works without external dependencies
- **JavaScript-free fallback**: Diagrams display even if JavaScript is disabled
- **Faster initial load**: No client-side rendering delay
- **Better SEO**: Search engines can index diagram content
- **Improved accessibility**: Static SVG with proper ARIA labels

## Generation

SVG files are automatically generated during the CI/CD build process by the [`scripts/render_mermaid.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/scripts/render_mermaid.py) script, which:

1. Scans all markdown files in [`docs/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/docs) for mermaid code blocks
2. Extracts the diagram code
3. Uses `@mermaid-js/mermaid-cli` (mmdc) to render each diagram to SVG
4. Saves files with reproducible names based on content hash

## File Naming

Files are named using the pattern: `{source-file}-{hash}.svg`

- `source-file`: The markdown file name (e.g., `pipeline`, `index`)
- `hash`: 8-character MD5 hash of the diagram code (ensures uniqueness)

## Usage

These SVG files are currently generated but not yet integrated into the documentation display. They are available for:

- Manual reference
- Offline documentation
- Future integration as `<noscript>` fallbacks
- Static documentation builds

## Updating

SVG files are regenerated on each documentation build. If you modify a mermaid diagram in the markdown source, the corresponding SVG will be automatically updated (with a new hash if the content changed).

## Local Generation

To generate SVG files locally:

```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Run the rendering script
python scripts/render_mermaid.py
```

The script uses [`puppeteer-config.json`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/puppeteer-config.json) for Puppeteer configuration (required for CI environments).
