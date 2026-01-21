#!/usr/bin/env python3
"""
Convert Markdown files to PDF using markdown + pdfkit or browser printing.

This script provides multiple methods to convert markdown to PDF:
1. Using pandoc + HTML + browser
2. Using markdown library to HTML, then print to PDF
"""

import subprocess
import sys
import os
from pathlib import Path
import markdown
import webbrowser
import tempfile


def convert_md_to_html(md_path: Path, output_path: Path = None) -> Path:
    """Convert markdown file to HTML with styling."""
    
    if output_path is None:
        output_path = md_path.with_suffix('.html')
    
    # Read markdown content
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert to HTML with extensions
    html_content = markdown.markdown(
        md_content,
        extensions=[
            'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            'markdown.extensions.codehilite',
            'markdown.extensions.toc',
            'markdown.extensions.meta',
        ]
    )
    
    # Wrap in HTML document with styling
    html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{md_path.stem}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px 40px;
            color: #333;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }}
        h1 {{ font-size: 2.2em; border-bottom: 2px solid #3498db; padding-bottom: 0.3em; }}
        h2 {{ font-size: 1.8em; border-bottom: 1px solid #bdc3c7; padding-bottom: 0.2em; }}
        h3 {{ font-size: 1.4em; }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
        }}
        pre {{
            background-color: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
            color: inherit;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 1em 0;
            padding: 0.5em 1em;
            background-color: #f9f9f9;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        ul, ol {{
            margin: 0.5em 0;
            padding-left: 2em;
        }}
        li {{
            margin: 0.3em 0;
        }}
        hr {{
            border: none;
            border-top: 1px solid #bdc3c7;
            margin: 2em 0;
        }}
        @media print {{
            body {{
                max-width: 100%;
                padding: 0;
            }}
            pre {{
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_document)
    
    print(f"✓ Created HTML: {output_path}")
    return output_path


def open_for_print(html_path: Path):
    """Open HTML file in browser for printing to PDF."""
    print(f"\n→ Opening {html_path} in browser...")
    print("  Use Ctrl+P (or Cmd+P on Mac) to print to PDF")
    webbrowser.open(f'file://{html_path.absolute()}')


def main():
    """Main function to convert markdown files to PDF."""
    
    # Default to README.md if no argument provided
    if len(sys.argv) > 1:
        md_file = Path(sys.argv[1])
    else:
        md_file = Path(__file__).parent / "README.md"
    
    if not md_file.exists():
        print(f"Error: File not found: {md_file}")
        sys.exit(1)
    
    print(f"Converting: {md_file}")
    print("=" * 50)
    
    # Try pandoc first if available
    try:
        # Check if pandoc is installed
        result = subprocess.run(['pandoc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Pandoc detected - using pandoc for conversion")
            
            # Output paths
            output_dir = md_file.parent / "docs" if md_file.name == "README.md" else md_file.parent
            html_output = output_dir / f"{md_file.stem}.html"
            
            # Convert to HTML with pandoc (better markdown support)
            pandoc_cmd = [
                'pandoc', str(md_file),
                '-o', str(html_output),
                '-s',  # standalone
                '--toc',  # table of contents
                '--toc-depth=3',
                '-c', 'https://cdn.jsdelivr.net/npm/github-markdown-css@5/github-markdown.min.css',
                '--metadata', f'title={md_file.stem}'
            ]
            
            result = subprocess.run(pandoc_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Created HTML with pandoc: {html_output}")
                open_for_print(html_output)
                return
            else:
                print(f"Pandoc conversion failed: {result.stderr}")
    except FileNotFoundError:
        print("Pandoc not found, using Python markdown library")
    
    # Fallback to Python markdown
    output_dir = md_file.parent / "docs" if md_file.name == "README.md" else md_file.parent
    html_output = output_dir / f"{md_file.stem}.html"
    
    html_path = convert_md_to_html(md_file, html_output)
    open_for_print(html_path)
    
    print("\n" + "=" * 50)
    print("To save as PDF:")
    print("  1. Press Ctrl+P in your browser")
    print("  2. Select 'Save as PDF' as the printer")
    print("  3. Click Save")


if __name__ == "__main__":
    main()
