#!/usr/bin/env python3
"""
Convert Markdown to DOCX format.
"""

import re
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def add_hyperlink(paragraph, text, url):
    """Add a hyperlink to a paragraph."""
    part = paragraph.part
    r_id = part.relate_to(url, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', is_external=True)

    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)

    new_run = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')

    # Add blue color and underline
    color = OxmlElement('w:color')
    color.set(qn('w:val'), '0000FF')
    rPr.append(color)

    u = OxmlElement('w:u')
    u.set(qn('w:val'), 'single')
    rPr.append(u)

    new_run.append(rPr)
    text_elem = OxmlElement('w:t')
    text_elem.text = text
    new_run.append(text_elem)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)

    return hyperlink


def process_inline_formatting(paragraph, text):
    """Process inline markdown formatting like bold, italic, code, links."""
    # Pattern for links: [text](url)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    # Pattern for bold: **text** or __text__
    bold_pattern = r'\*\*([^*]+)\*\*|__([^_]+)__'
    # Pattern for italic: *text* or _text_
    italic_pattern = r'(?<!\*)\*(?!\*)([^*]+)\*(?!\*)|(?<!_)_(?!_)([^_]+)_(?!_)'
    # Pattern for inline code: `code`
    code_pattern = r'`([^`]+)`'

    # Simple approach: just add text with basic formatting
    parts = []
    last_end = 0

    # Find all links first
    for match in re.finditer(link_pattern, text):
        if match.start() > last_end:
            parts.append(('text', text[last_end:match.start()]))
        parts.append(('link', match.group(1), match.group(2)))
        last_end = match.end()

    if last_end < len(text):
        parts.append(('text', text[last_end:]))

    for part in parts:
        if part[0] == 'text':
            content = part[1]
            # Process bold
            content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
            content = re.sub(r'__([^_]+)__', r'\1', content)
            # Process italic
            content = re.sub(r'(?<!\*)\*(?!\*)([^*]+)\*(?!\*)', r'\1', content)
            # Process code
            content = re.sub(r'`([^`]+)`', r'\1', content)
            if content:
                paragraph.add_run(content)
        elif part[0] == 'link':
            add_hyperlink(paragraph, part[1], part[2])


def convert_md_to_docx(md_path, docx_path):
    """Convert a markdown file to DOCX format."""

    # Read markdown content
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Create document
    doc = Document()

    # Set up styles
    styles = doc.styles

    # Process content line by line
    lines = content.split('\n')
    i = 0
    in_code_block = False
    code_block_content = []
    in_table = False
    table_rows = []

    while i < len(lines):
        line = lines[i]

        # Handle code blocks
        if line.startswith('```'):
            if in_code_block:
                # End code block
                in_code_block = False
                if code_block_content:
                    para = doc.add_paragraph()
                    para.style = 'No Spacing'
                    for code_line in code_block_content:
                        run = para.add_run(code_line + '\n')
                        run.font.name = 'Courier New'
                        run.font.size = Pt(9)
                    code_block_content = []
            else:
                # Start code block
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_block_content.append(line)
            i += 1
            continue

        # Handle tables
        if '|' in line and not line.startswith('```'):
            # Check if it's a table row
            stripped = line.strip()
            if stripped.startswith('|') and stripped.endswith('|'):
                if not in_table:
                    in_table = True
                    table_rows = []

                # Skip separator rows (|---|---|)
                if re.match(r'^\|[\s\-:|]+\|$', stripped):
                    i += 1
                    continue

                # Parse table row
                cells = [cell.strip() for cell in stripped.split('|')[1:-1]]
                table_rows.append(cells)
                i += 1
                continue

        # End table if we were in one
        if in_table and ('|' not in line or line.startswith('```')):
            in_table = False
            if table_rows:
                # Create table
                num_cols = max(len(row) for row in table_rows)
                table = doc.add_table(rows=len(table_rows), cols=num_cols)
                table.style = 'Table Grid'

                for row_idx, row_data in enumerate(table_rows):
                    for col_idx, cell_data in enumerate(row_data):
                        if col_idx < num_cols:
                            cell = table.rows[row_idx].cells[col_idx]
                            # Clean up markdown formatting
                            cell_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', cell_data)
                            cell_text = re.sub(r'`([^`]+)`', r'\1', cell_text)
                            cell.text = cell_text

                doc.add_paragraph()  # Add space after table
                table_rows = []

        # Handle horizontal rules
        if line.strip() == '---' or line.strip() == '***':
            doc.add_paragraph('_' * 50)
            i += 1
            continue

        # Handle headings
        if line.startswith('#'):
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                level = len(match.group(1))
                text = match.group(2)

                if level == 1:
                    para = doc.add_heading(text, level=0)
                else:
                    para = doc.add_heading(text, level=min(level, 9))

                i += 1
                continue

        # Handle bullet lists
        if line.strip().startswith('- ') or line.strip().startswith('* '):
            text = line.strip()[2:]
            para = doc.add_paragraph(style='List Bullet')
            process_inline_formatting(para, text)
            i += 1
            continue

        # Handle numbered lists
        match = re.match(r'^(\d+)\.\s+(.+)$', line.strip())
        if match:
            text = match.group(2)
            para = doc.add_paragraph(style='List Number')
            process_inline_formatting(para, text)
            i += 1
            continue

        # Handle regular paragraphs
        if line.strip():
            para = doc.add_paragraph()
            process_inline_formatting(para, line)

        i += 1

    # Handle any remaining table
    if in_table and table_rows:
        num_cols = max(len(row) for row in table_rows)
        table = doc.add_table(rows=len(table_rows), cols=num_cols)
        table.style = 'Table Grid'

        for row_idx, row_data in enumerate(table_rows):
            for col_idx, cell_data in enumerate(row_data):
                if col_idx < num_cols:
                    cell = table.rows[row_idx].cells[col_idx]
                    cell_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', cell_data)
                    cell_text = re.sub(r'`([^`]+)`', r'\1', cell_text)
                    cell.text = cell_text

    # Save document
    doc.save(docx_path)
    print(f"Document saved to: {docx_path}")


if __name__ == '__main__':
    md_path = Path('/home/user/lowcost-star-tracker/docs/LowCost_StarTracker_Technical_Paper.md')
    docx_path = Path('/home/user/lowcost-star-tracker/docs/LowCost_StarTracker_Technical_Paper.docx')

    convert_md_to_docx(md_path, docx_path)
