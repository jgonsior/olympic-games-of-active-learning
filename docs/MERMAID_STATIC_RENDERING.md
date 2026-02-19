# Mermaid Static Rendering

## Overview

OGAL documentation uses **two rendering methods** for Mermaid diagrams:

- **Client-side** (default): Interactive diagrams via the `mkdocs-mermaid2-plugin`
- **Server-side** (fallback): Pre-rendered SVGs generated during CI/CD build

## How It Works

1. [`scripts/render_mermaid.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/scripts/render_mermaid.py) scans markdown files for mermaid code blocks
2. Renders each diagram to SVG using `mmdc` (mermaid-cli)
3. Saves files to `docs/images/mermaid/` with content-hash filenames (`{source}-{hash}.svg`)

The CI/CD pipeline installs mermaid-cli, runs the script, then builds the docs.

## Local Generation

```bash
npm install -g @mermaid-js/mermaid-cli
python scripts/render_mermaid.py
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `mmdc command not found` | `npm install -g @mermaid-js/mermaid-cli` |
| `Failed to launch browser` | Ensure `puppeteer-config.json` has no-sandbox flags |
| `Timeout rendering` | Check diagram syntax for errors |
