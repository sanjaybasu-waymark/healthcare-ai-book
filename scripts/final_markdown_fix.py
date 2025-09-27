#!/usr/bin/env python3
"""
Final targeted fix for markdown rendering issues
"""

import os
import re

def fix_markdown_in_file(filepath):
    """Fix markdown rendering in a single file"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix escaped headers - be very specific
    content = re.sub(r'\\#\s+', '# ', content)
    content = re.sub(r'\\##\s+', '## ', content)
    content = re.sub(r'\\###\s+', '### ', content)
    content = re.sub(r'\\####\s+', '#### ', content)
    
    # Fix any remaining escaped markdown
    content = re.sub(r'\\([#*_])', r'\1', content)
    
    # Remove line numbers from code blocks more aggressively
    def clean_code_block(match):
        start = match.group(1)
        code_content = match.group(2)
        end = match.group(3)
        
        # Split into lines and clean each line
        lines = code_content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove line numbers at the start of lines (more patterns)
            cleaned_line = re.sub(r'^\s*\d+\s*', '', line)
            cleaned_line = re.sub(r'^\s*\d+$', '', cleaned_line)
            cleaned_lines.append(cleaned_line)
        
        cleaned_code = '\n'.join(cleaned_lines)
        return start + cleaned_code + end
    
    # Apply code block cleaning
    code_block_pattern = r'(```[a-zA-Z]*\n)(.*?)(```)'
    content = re.sub(code_block_pattern, clean_code_block, content, flags=re.DOTALL)
    
    # Also remove standalone line numbers outside code blocks
    content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
    
    # Only write if content changed
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    print("🔧 FINAL MARKDOWN RENDERING FIX")
    print("=" * 40)
    
    fixed_count = 0
    
    # Fix all chapter files
    chapter_files = [f for f in os.listdir("_chapters") if f.endswith(".md")]
    
    for chapter_file in sorted(chapter_files):
        filepath = f"_chapters/{chapter_file}"
        if fix_markdown_in_file(filepath):
            print(f"✅ Fixed {chapter_file}")
            fixed_count += 1
        else:
            print(f"⚪ No changes needed for {chapter_file}")
    
    print(f"\n📊 Summary: Fixed {fixed_count} files")
    
    if fixed_count > 0:
        print("\n🔄 Changes made - commit and push to deploy")
    else:
        print("\n✅ All files already properly formatted")

if __name__ == "__main__":
    main()
