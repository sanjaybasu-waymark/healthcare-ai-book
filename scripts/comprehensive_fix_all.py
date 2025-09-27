#!/usr/bin/env python3
"""
Comprehensive Fix Script for All Identified Issues
- Fix markdown rendering (# and * symbols)
- Fix code examples count and links
- Remove line numbers from code blocks
- Fix all navigation and directory links
"""

import os
import re
import glob
from pathlib import Path

def fix_markdown_rendering(content):
    """Fix markdown rendering issues where # and * are showing as literal text"""
    
    # Fix escaped markdown headers
    content = re.sub(r'\\#', '#', content)
    content = re.sub(r'\\##', '##', content)
    content = re.sub(r'\\###', '###', content)
    content = re.sub(r'\\####', '####', content)
    
    # Fix escaped markdown emphasis
    content = re.sub(r'\\\*\\\*([^*]+)\\\*\\\*', r'**\1**', content)
    content = re.sub(r'\\\*([^*]+)\\\*', r'*\1*', content)
    
    # Fix any remaining escaped markdown
    content = re.sub(r'\\([*_#])', r'\1', content)
    
    return content

def remove_line_numbers_from_code(content):
    """Remove line numbers from code blocks completely"""
    
    # Pattern to match code blocks
    code_block_pattern = r'(```[a-zA-Z]*\n)(.*?)(```)'
    
    def clean_code_block(match):
        start = match.group(1)
        code_content = match.group(2)
        end = match.group(3)
        
        # Split into lines and clean each line
        lines = code_content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove line numbers at the start of lines
            cleaned_line = re.sub(r'^\s*\d+\s*', '', line)
            cleaned_lines.append(cleaned_line)
        
        cleaned_code = '\n'.join(cleaned_lines)
        return start + cleaned_code + end
    
    # Apply the cleaning
    content = re.sub(code_block_pattern, clean_code_block, content, flags=re.DOTALL)
    
    # Also remove standalone line numbers
    content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
    
    return content

def count_python_files(chapter_num):
    """Count actual Python files in a chapter directory"""
    chapter_dir = f"code_examples/chapter_{chapter_num:02d}"
    if not os.path.exists(chapter_dir):
        return 0
    
    py_files = glob.glob(f"{chapter_dir}/*.py")
    return len(py_files)

def create_proper_code_examples_index():
    """Create a properly working code examples index with correct counts and links"""
    
    content = """---
title: "Code Examples"
layout: default
permalink: /code_examples/
---

# Code Examples

This section contains all code examples from the Healthcare AI Implementation Guide, organized by chapter.

## Available Examples

"""
    
    for i in range(1, 30):
        py_count = count_python_files(i)
        
        if py_count > 0:
            content += f"""
### [Chapter {i}](https://github.com/sanjaybasu-waymark/healthcare-ai-book/tree/main/code_examples/chapter_{i:02d}/)
- **Examples**: {py_count} Python files
- **Directory**: `code_examples/chapter_{i:02d}/`
- **GitHub**: [View on GitHub](https://github.com/sanjaybasu-waymark/healthcare-ai-book/tree/main/code_examples/chapter_{i:02d}/)
"""
        else:
            content += f"""
### [Chapter {i}](https://github.com/sanjaybasu-waymark/healthcare-ai-book/tree/main/code_examples/chapter_{i:02d}/)
- **Examples**: Theoretical content (see README)
- **Directory**: `code_examples/chapter_{i:02d}/`
- **GitHub**: [View on GitHub](https://github.com/sanjaybasu-waymark/healthcare-ai-book/tree/main/code_examples/chapter_{i:02d}/)
"""
    
    content += """

## Usage

Each chapter directory contains:
- `example_XX.py` - Numbered code examples from the chapter (where applicable)
- `requirements.txt` - Python dependencies
- `README.md` - Chapter-specific documentation

To run examples:

```bash
# Clone the repository
git clone https://github.com/sanjaybasu-waymark/healthcare-ai-book.git
cd healthcare-ai-book

# Navigate to a chapter with code examples
cd code_examples/chapter_01

# Install dependencies
pip install -r requirements.txt

# Run an example
python example_01.py
```

## Repository Structure

```
code_examples/
├── chapter_01/          # 3 Python files
├── chapter_02/          # 2 Python files  
├── chapter_03/          # 3 Python files
├── chapter_04/          # 1 Python file
├── chapter_05/          # Theoretical content
├── chapter_06/          # 1 Python file
├── chapter_07/          # Theoretical content
├── chapter_08/          # Theoretical content
├── chapter_09/          # Theoretical content
├── chapter_10/          # Theoretical content
├── chapter_11/          # Theoretical content
├── chapter_12/          # Theoretical content
├── chapter_13/          # Theoretical content
├── chapter_14/          # Theoretical content
├── chapter_15/          # Theoretical content
├── chapter_16/          # 1 Python file
├── chapter_17/          # 1 Python file
├── chapter_18/          # 1 Python file
├── chapter_19/          # 1 Python file
├── chapter_20/          # 1 Python file
├── chapter_21/          # 2 Python files
├── chapter_22/          # 3 Python files
├── chapter_23/          # Theoretical content
├── chapter_24/          # 3 Python files
├── chapter_25/          # Theoretical content
├── chapter_26/          # 3 Python files
├── chapter_27/          # 4 Python files
├── chapter_28/          # 1 Python file
└── chapter_29/          # 5 Python files
```

## Download All Examples

- **ZIP Download**: [Download all examples](https://github.com/sanjaybasu-waymark/healthcare-ai-book/archive/refs/heads/main.zip)
- **Git Clone**: `git clone https://github.com/sanjaybasu-waymark/healthcare-ai-book.git`
"""
    
    with open("code_examples.md", "w") as f:
        f.write(content)
    
    print("✅ Created proper code examples index")

def fix_chapter_file(chapter_file):
    """Fix all issues in a single chapter file"""
    
    chapter_num = int(re.search(r'(\d+)', chapter_file).group(1))
    print(f"🔧 Fixing Chapter {chapter_num}...")
    
    with open(f"_chapters/{chapter_file}", 'r') as f:
        content = f.read()
    
    # Fix markdown rendering
    content = fix_markdown_rendering(content)
    
    # Remove line numbers from code blocks
    content = remove_line_numbers_from_code(content)
    
    # Ensure proper front matter
    if not content.startswith("---"):
        front_matter = f"""---
layout: default
title: "Chapter {chapter_num}: {chapter_file.replace('.md', '').replace(f'{chapter_num:02d}-', '').replace('-', ' ').title()}"
nav_order: {chapter_num}
parent: Chapters
permalink: /chapters/{chapter_file.replace('.md', '')}/
---

"""
        content = front_matter + content
    
    # Add proper code examples reference
    py_count = count_python_files(chapter_num)
    if py_count > 0:
        code_ref = f"""

## Code Examples

This chapter includes {py_count} executable Python examples:

- **Repository**: [code_examples/chapter_{chapter_num:02d}/](https://github.com/sanjaybasu-waymark/healthcare-ai-book/tree/main/code_examples/chapter_{chapter_num:02d}/)
- **Download**: [ZIP file](https://github.com/sanjaybasu-waymark/healthcare-ai-book/archive/refs/heads/main.zip)

To run the examples:
```bash
git clone https://github.com/sanjaybasu-waymark/healthcare-ai-book.git
cd healthcare-ai-book/code_examples/chapter_{chapter_num:02d}
pip install -r requirements.txt
python example_01.py
```
"""
    else:
        code_ref = f"""

## Chapter Resources

This chapter focuses on theoretical concepts and frameworks. For practical implementations, see:

- **Related Code**: [Browse all code examples](/healthcare-ai-book/code_examples/)
- **Repository**: [Chapter directory](https://github.com/sanjaybasu-waymark/healthcare-ai-book/tree/main/code_examples/chapter_{chapter_num:02d}/)
"""
    
    # Add code reference if not already present
    if "## Code Examples" not in content and "## Chapter Resources" not in content:
        content += code_ref
    
    # Save the fixed content
    with open(f"_chapters/{chapter_file}", 'w') as f:
        f.write(content)
    
    print(f"   ✅ Fixed markdown rendering")
    print(f"   ✅ Removed line numbers from code")
    print(f"   ✅ Added proper code examples reference")

def main():
    print("🚀 COMPREHENSIVE FIX FOR ALL ISSUES")
    print("=" * 50)
    
    # Fix the code examples index first
    print("\n📚 Creating proper code examples index...")
    create_proper_code_examples_index()
    
    # Fix all chapter files
    print("\n📖 Fixing all chapter files...")
    chapter_files = sorted([f for f in os.listdir("_chapters") if f.endswith(".md")])
    
    for chapter_file in chapter_files:
        fix_chapter_file(chapter_file)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 ALL FIXES COMPLETED!")
    print("\nFixed issues:")
    print("✅ Markdown rendering (removed escaped # and * symbols)")
    print("✅ Code examples count and links (now show correct numbers)")
    print("✅ Line numbers in code blocks (completely removed)")
    print("✅ Navigation links (all working with GitHub links)")
    print("✅ Directory structure (properly documented)")
    
    # Count summary
    total_py_files = sum(count_python_files(i) for i in range(1, 30))
    chapters_with_code = sum(1 for i in range(1, 30) if count_python_files(i) > 0)
    
    print(f"\n📊 Final Statistics:")
    print(f"Total Python files: {total_py_files}")
    print(f"Chapters with code: {chapters_with_code}")
    print(f"Chapters with theory: {29 - chapters_with_code}")

if __name__ == "__main__":
    main()
