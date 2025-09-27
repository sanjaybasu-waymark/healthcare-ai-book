#!/usr/bin/env python3
"""
Fix bibliography formatting issues where code blocks and references merge together
"""

import os
import re

def fix_bibliography_formatting(filepath):
    """Fix bibliography formatting in a single chapter file"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Find incomplete code blocks (missing closing ```)
    # Count opening and closing code block markers
    opening_markers = content.count('```')
    if opening_markers % 2 != 0:
        # Odd number means there's an unclosed code block
        print(f"   ⚠️  Found unclosed code block in {os.path.basename(filepath)}")
        
        # Find the last ``` and add a closing one if needed
        last_marker_pos = content.rfind('```')
        if last_marker_pos != -1:
            # Check if there's content after the last marker that looks like bibliography
            after_marker = content[last_marker_pos + 3:]
            if any(keyword in after_marker.lower() for keyword in ['references', 'bibliography', 'citations', '##']):
                # Insert closing ``` before the bibliography section
                content = content[:last_marker_pos + 3] + '\n```\n\n' + after_marker
                print(f"   ✅ Added missing closing ``` before bibliography")
    
    # Ensure proper separation between code examples and bibliography
    # Look for patterns where code examples run into bibliography
    patterns_to_fix = [
        # Code block followed immediately by ## References or similar
        (r'```\s*\n(#+\s*(References|Bibliography|Citations))', r'```\n\n\1'),
        
        # Code examples section followed immediately by references
        (r'(## Code Examples.*?)\n(#+\s*(References|Bibliography))', r'\1\n\n\2'),
        
        # Any section that runs into references without proper spacing
        (r'([^\n])\n(#+\s*(References|Bibliography|Citations))', r'\1\n\n\2'),
        
        # Fix incomplete code blocks at end of file
        (r'```\s*$', '```\n'),
    ]
    
    for pattern, replacement in patterns_to_fix:
        if re.search(pattern, content, re.MULTILINE | re.DOTALL):
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
            print(f"   ✅ Fixed bibliography separation pattern")
    
    # Ensure bibliography sections have proper formatting
    # Add proper spacing before bibliography headers
    content = re.sub(r'\n(#+\s*(References|Bibliography|Citations))', r'\n\n\1', content)
    
    # Ensure code examples section is properly formatted
    if '## Code Examples' in content:
        # Make sure there's proper spacing before and after
        content = re.sub(r'\n(## Code Examples)', r'\n\n\1', content)
        content = re.sub(r'(## Code Examples.*?)\n([^#\n])', r'\1\n\n\2', content, flags=re.DOTALL)
    
    # Clean up excessive newlines (more than 3 in a row)
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    
    # Only write if content changed
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def check_bibliography_issues(filepath):
    """Check for potential bibliography formatting issues"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    
    # Check for unclosed code blocks
    opening_markers = content.count('```')
    if opening_markers % 2 != 0:
        issues.append("Unclosed code block")
    
    # Check for bibliography sections without proper spacing
    if re.search(r'[^\n]\n#+\s*(References|Bibliography|Citations)', content):
        issues.append("Bibliography without proper spacing")
    
    # Check for code running into bibliography
    if re.search(r'```\s*\n#+\s*(References|Bibliography)', content):
        issues.append("Code block runs into bibliography")
    
    return issues

def main():
    print("🔧 FIXING BIBLIOGRAPHY FORMATTING ISSUES")
    print("=" * 50)
    
    # First, check all chapters for issues
    print("\n📋 Checking all chapters for bibliography issues...")
    
    chapter_files = [f for f in os.listdir("_chapters") if f.endswith(".md")]
    chapters_with_issues = []
    
    for chapter_file in sorted(chapter_files):
        filepath = f"_chapters/{chapter_file}"
        issues = check_bibliography_issues(filepath)
        
        if issues:
            chapters_with_issues.append((chapter_file, issues))
            print(f"❌ {chapter_file}: {', '.join(issues)}")
        else:
            print(f"✅ {chapter_file}: No issues found")
    
    if not chapters_with_issues:
        print("\n🎉 No bibliography formatting issues found!")
        return
    
    print(f"\n🔧 Fixing {len(chapters_with_issues)} chapters with issues...")
    
    fixed_count = 0
    for chapter_file, issues in chapters_with_issues:
        filepath = f"_chapters/{chapter_file}"
        print(f"\n📖 Fixing {chapter_file}...")
        
        if fix_bibliography_formatting(filepath):
            fixed_count += 1
            print(f"   ✅ Fixed formatting issues")
        else:
            print(f"   ⚪ No changes needed")
    
    print(f"\n📊 Summary:")
    print(f"Chapters checked: {len(chapter_files)}")
    print(f"Chapters with issues: {len(chapters_with_issues)}")
    print(f"Chapters fixed: {fixed_count}")
    
    if fixed_count > 0:
        print("\n🔄 Changes made - commit and push to deploy fixes")
    else:
        print("\n✅ All bibliography formatting is correct")

if __name__ == "__main__":
    main()
