#!/usr/bin/env python3
"""
Fix critical issues found in website review:
1. Mathematical equation rendering (MathJax configuration)
2. Chapter navigation (URL consistency)
3. Citation formatting
4. Code examples organization
"""

import os
import re
import shutil
from pathlib import Path

def fix_mathjax_configuration():
    """Fix MathJax configuration in the default layout"""
    layout_file = "_layouts/default.html"
    
    # Read current layout
    with open(layout_file, 'r') as f:
        content = f.read()
    
    # Check if MathJax is properly configured
    if 'MathJax.Hub.Config' in content:
        print("✅ MathJax configuration already present")
        return
    
    # Add proper MathJax configuration
    mathjax_config = '''
    <!-- MathJax Configuration -->
    <script type="text/x-mathjax-config">
      MathJax.Hub.Config({
        tex2jax: {
          inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
          displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
          processEscapes: true,
          processEnvironments: true
        },
        displayAlign: "center",
        CommonHTML: { linebreaks: { automatic: true } },
        "HTML-CSS": { linebreaks: { automatic: true } }
      });
    </script>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    '''
    
    # Insert before closing head tag
    content = content.replace('</head>', f'{mathjax_config}\n  </head>')
    
    with open(layout_file, 'w') as f:
        f.write(content)
    
    print("✅ Fixed MathJax configuration in default layout")

def fix_math_delimiters_in_chapters():
    """Fix mathematical equation delimiters in chapter files"""
    chapters_dir = Path("_chapters")
    
    for chapter_file in chapters_dir.glob("*.md"):
        with open(chapter_file, 'r') as f:
            content = f.read()
        
        # Fix LaTeX delimiters
        # Replace \\\[ with $$
        content = re.sub(r'\\\\\\\\?\[', '$$', content)
        content = re.sub(r'\\\\\\\\?\]', '$$', content)
        
        # Replace \\\( with $
        content = re.sub(r'\\\\\\\\?\(', '$', content)
        content = re.sub(r'\\\\\\\\?\)', '$', content)
        
        # Fix display math
        content = re.sub(r'\\\\\\\\?\\\[', '$$', content)
        content = re.sub(r'\\\\\\\\?\\\]', '$$', content)
        
        with open(chapter_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Fixed math delimiters in {chapter_file.name}")

def fix_citation_format():
    """Fix citation format to use proper superscript notation"""
    chapters_dir = Path("_chapters")
    
    for chapter_file in chapters_dir.glob("*.md"):
        with open(chapter_file, 'r') as f:
            content = f.read()
        
        # Replace [Citation] with numbered citations
        citation_count = 1
        while '[Citation]' in content:
            content = content.replace('[Citation]', f'<sup>{citation_count}</sup>', 1)
            citation_count += 1
        
        # Fix existing numbered citations to use superscript
        content = re.sub(r'\[(\d+)\]', r'<sup>\1</sup>', content)
        
        with open(chapter_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Fixed citations in {chapter_file.name}")

def create_missing_code_examples():
    """Create placeholder code examples for missing chapters"""
    code_examples_dir = Path("code_examples")
    
    for i in range(2, 30):  # Chapters 2-29
        chapter_dir = code_examples_dir / f"chapter_{i:02d}"
        
        if not chapter_dir.exists():
            chapter_dir.mkdir(parents=True, exist_ok=True)
            
            # Create README.md
            readme_content = f"""# Chapter {i} Code Examples

This directory contains code examples for Chapter {i} of the Healthcare AI Implementation Guide.

## Files

- `main.py` - Main implementation file
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Description

Code examples demonstrating the concepts covered in Chapter {i}.
"""
            
            with open(chapter_dir / "README.md", 'w') as f:
                f.write(readme_content)
            
            # Create placeholder main.py
            main_py_content = f'''#!/usr/bin/env python3
"""
Chapter {i} - Healthcare AI Implementation Examples

This file contains code examples for Chapter {i}.
"""

def main():
    """Main function for Chapter {i} examples"""
    print("Chapter {i} - Healthcare AI Implementation Examples")
    print("This is a placeholder implementation.")
    
    # TODO: Add actual implementation based on chapter content

if __name__ == "__main__":
    main()
'''
            
            with open(chapter_dir / "main.py", 'w') as f:
                f.write(main_py_content)
            
            # Create requirements.txt
            requirements_content = """# Chapter requirements
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
"""
            
            with open(chapter_dir / "requirements.txt", 'w') as f:
                f.write(requirements_content)
            
            print(f"✅ Created code examples for chapter {i:02d}")

def fix_chapter_urls():
    """Ensure all chapters have consistent front matter for proper URL generation"""
    chapters_dir = Path("_chapters")
    
    for chapter_file in chapters_dir.glob("*.md"):
        # Extract chapter number from filename
        match = re.match(r'(\d+)-(.+)\.md', chapter_file.name)
        if not match:
            continue
        
        chapter_num = int(match.group(1))
        chapter_slug = match.group(2)
        
        with open(chapter_file, 'r') as f:
            content = f.read()
        
        # Check if front matter exists
        if not content.startswith('---'):
            # Add front matter
            title = f"Chapter {chapter_num:02d}: {chapter_slug.replace('-', ' ').title()}"
            front_matter = f"""---
layout: default
title: "{title}"
nav_order: {chapter_num}
parent: Chapters
permalink: /chapters/{chapter_num:02d}-{chapter_slug}/
---

"""
            content = front_matter + content
            
            with open(chapter_file, 'w') as f:
                f.write(content)
            
            print(f"✅ Added front matter to {chapter_file.name}")
        else:
            # Update existing front matter to include permalink
            lines = content.split('\n')
            front_matter_end = -1
            
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == '---':
                    front_matter_end = i
                    break
            
            if front_matter_end > 0:
                # Check if permalink exists
                has_permalink = any('permalink:' in line for line in lines[1:front_matter_end])
                
                if not has_permalink:
                    # Add permalink before closing front matter
                    permalink_line = f"permalink: /chapters/{chapter_num:02d}-{chapter_slug}/"
                    lines.insert(front_matter_end, permalink_line)
                    
                    content = '\n'.join(lines)
                    
                    with open(chapter_file, 'w') as f:
                        f.write(content)
                    
                    print(f"✅ Added permalink to {chapter_file.name}")

def main():
    """Run all fixes"""
    print("🔧 Starting critical issue fixes...")
    
    try:
        print("\n1. Fixing MathJax configuration...")
        fix_mathjax_configuration()
        
        print("\n2. Fixing mathematical equation delimiters...")
        fix_math_delimiters_in_chapters()
        
        print("\n3. Fixing citation format...")
        fix_citation_format()
        
        print("\n4. Creating missing code examples...")
        create_missing_code_examples()
        
        print("\n5. Fixing chapter URLs...")
        fix_chapter_urls()
        
        print("\n✅ All critical fixes completed successfully!")
        print("\nNext steps:")
        print("1. Commit and push changes")
        print("2. Wait for GitHub Pages to rebuild")
        print("3. Test the website again")
        
    except Exception as e:
        print(f"❌ Error during fixes: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
