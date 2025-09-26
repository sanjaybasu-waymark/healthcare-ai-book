#!/usr/bin/env python3
"""
Comprehensive Chapter Optimization Script
Fixes math rendering, code blocks, navigation, and syncs code examples
"""

import os
import re
import shutil
from pathlib import Path

def fix_math_rendering(content):
    """Fix math rendering issues including macro parameter character errors"""
    
    # Fix macro parameter character # in math mode
    content = re.sub(r'\\#', r'\\text{#}', content)
    content = re.sub(r'(?<!\\)#(?![a-zA-Z])', r'\\#', content)
    
    # Fix common LaTeX issues
    content = re.sub(r'\\textbf\{([^}]+)\}', r'\\mathbf{\\1}', content)
    content = re.sub(r'\\text\{([^}]+)\}', r'\\mathrm{\\1}', content)
    
    # Ensure proper math delimiters
    content = re.sub(r'\\\[([^\\]+)\\\]', r'$$\\1$$', content)
    content = re.sub(r'\\\(([^\\]+)\\\)', r'$\\1$', content)
    
    return content

def fix_code_blocks(content):
    """Fix code block formatting and remove line numbers"""
    
    # Pattern to match code blocks with line numbers
    code_block_pattern = r'```([a-zA-Z]*)\n((?:\s*\d+\s*.*\n?)+)```'
    
    def clean_code_block(match):
        language = match.group(1)
        code_lines = match.group(2).strip().split('\n')
        
        # Remove line numbers from each line
        cleaned_lines = []
        for line in code_lines:
            # Remove leading line numbers (e.g., "    1    ", "   10    ", etc.)
            cleaned_line = re.sub(r'^\s*\d+\s+', '', line)
            cleaned_lines.append(cleaned_line)
        
        # Join back and create proper code block
        cleaned_code = '\n'.join(cleaned_lines)
        return f'```{language}\n{cleaned_code}\n```'
    
    # Apply the fix
    content = re.sub(code_block_pattern, clean_code_block, content, flags=re.MULTILINE | re.DOTALL)
    
    # Fix any remaining standalone line numbers
    content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
    
    return content

def extract_and_save_code_examples(content, chapter_num):
    """Extract code examples from chapter and save to code_examples directory"""
    
    chapter_dir = f"code_examples/chapter_{chapter_num:02d}"
    os.makedirs(chapter_dir, exist_ok=True)
    
    # Find all code blocks
    code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
    
    example_count = 1
    for code_block in code_blocks:
        # Clean the code block
        cleaned_code = re.sub(r'^\s*\d+\s+', '', code_block, flags=re.MULTILINE)
        
        # Save to file
        filename = f"{chapter_dir}/example_{example_count:02d}.py"
        with open(filename, 'w') as f:
            f.write(f'"""\nChapter {chapter_num} - Example {example_count}\nExtracted from Healthcare AI Implementation Guide\n"""\n\n')
            f.write(cleaned_code)
        
        print(f"   💾 Saved {filename}")
        example_count += 1
    
    # Create requirements.txt for each chapter
    requirements_file = f"{chapter_dir}/requirements.txt"
    if not os.path.exists(requirements_file):
        with open(requirements_file, 'w') as f:
            f.write("""# Chapter requirements
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
""")

def create_code_examples_index():
    """Create an index page for code examples"""
    
    index_content = """---
title: "Code Examples"
layout: default
permalink: /code_examples/
---

# Code Examples

This section contains all code examples from the Healthcare AI Implementation Guide, organized by chapter.

## Available Examples

"""
    
    for i in range(1, 30):
        chapter_dir = f"code_examples/chapter_{i:02d}"
        if os.path.exists(chapter_dir):
            # Count Python files
            py_files = [f for f in os.listdir(chapter_dir) if f.endswith('.py')]
            example_count = len(py_files)
            
            index_content += f"""
### [Chapter {i}](chapter_{i:02d}/)
- **Examples**: {example_count} Python files
- **Directory**: `code_examples/chapter_{i:02d}/`
"""
    
    index_content += """

## Usage

Each chapter directory contains:
- `example_XX.py` - Numbered code examples from the chapter
- `requirements.txt` - Python dependencies
- `README.md` - Chapter-specific documentation

To run examples:

```bash
cd code_examples/chapter_XX
pip install -r requirements.txt
python example_01.py
```

## Repository Structure

```
code_examples/
├── chapter_01/
│   ├── example_01.py
│   ├── example_02.py
│   ├── requirements.txt
│   └── README.md
├── chapter_02/
│   └── ...
└── ...
```
"""
    
    with open("code_examples.md", "w") as f:
        f.write(index_content)

def optimize_chapter(chapter_file):
    """Optimize a single chapter file"""
    
    chapter_num = int(re.search(r'(\d+)', chapter_file).group(1))
    print(f"\n📖 Optimizing Chapter {chapter_num}...")
    
    with open(f"_chapters/{chapter_file}", 'r') as f:
        content = f.read()
    
    # Fix math rendering
    print("   🧮 Fixing math rendering...")
    content = fix_math_rendering(content)
    
    # Fix code blocks
    print("   💻 Fixing code blocks...")
    content = fix_code_blocks(content)
    
    # Extract and save code examples
    print("   📁 Extracting code examples...")
    extract_and_save_code_examples(content, chapter_num)
    
    # Add code example references
    if f"code_examples/chapter_{chapter_num:02d}" in content:
        pass  # Already has reference
    else:
        # Add reference to code examples at the end
        content += f"""

## Code Examples

All code examples from this chapter are available in the repository:
- **Directory**: [`code_examples/chapter_{chapter_num:02d}/`](https://github.com/sanjaybasu-waymark/healthcare-ai-book/tree/main/code_examples/chapter_{chapter_num:02d}/)
- **Direct Download**: [ZIP file](https://github.com/sanjaybasu-waymark/healthcare-ai-book/archive/refs/heads/main.zip)

To use the examples:
```bash
git clone https://github.com/sanjaybasu-waymark/healthcare-ai-book.git
cd healthcare-ai-book/code_examples/chapter_{chapter_num:02d}
pip install -r requirements.txt
```
"""
    
    # Save optimized content
    with open(f"_chapters/{chapter_file}", 'w') as f:
        f.write(content)
    
    print(f"   ✅ Chapter {chapter_num} optimized")

def fix_navigation_links():
    """Fix all navigation links"""
    
    print("\n🔗 Fixing navigation links...")
    
    # Fix index.md
    with open("index.md", 'r') as f:
        content = f.read()
    
    # Fix Code Examples link
    content = re.sub(r'\[Code Examples\]\(\)', '[Code Examples](/healthcare-ai-book/code_examples/)', content)
    content = re.sub(r'\[GitHub repository\]\(\)', '[GitHub repository](https://github.com/sanjaybasu-waymark/healthcare-ai-book)', content)
    
    with open("index.md", 'w') as f:
        f.write(content)
    
    print("   ✅ Fixed index.md navigation")

def main():
    print("🚀 Starting Comprehensive Chapter Optimization...")
    print("=" * 60)
    
    # Fix navigation first
    fix_navigation_links()
    
    # Create code examples index
    print("\n📚 Creating code examples index...")
    create_code_examples_index()
    
    # Process all chapters
    chapter_files = sorted([f for f in os.listdir("_chapters") if f.endswith(".md")])
    
    for chapter_file in chapter_files:
        optimize_chapter(chapter_file)
    
    print("\n" + "=" * 60)
    print("🎉 Comprehensive optimization completed!")
    print("\nSummary:")
    print("✅ Fixed math rendering errors (macro parameter characters)")
    print("✅ Cleaned up code blocks (removed line numbers)")
    print("✅ Extracted code examples to repository")
    print("✅ Created code examples index page")
    print("✅ Fixed navigation links")
    print("✅ Added code example references to chapters")
    print("\nAll chapters are now optimized for display and usability!")

if __name__ == "__main__":
    main()
