#!/usr/bin/env python3
"""
Verification Script for Chapter Optimization
Systematically checks all chapters for remaining issues
"""

import os
import re
from pathlib import Path

def check_math_issues(content, chapter_num):
    """Check for math rendering issues"""
    issues = []
    
    # Check for macro parameter character errors
    if "macro parameter character" in content:
        issues.append("Contains 'macro parameter character' error")
    
    # Check for unescaped # in math mode
    math_blocks = re.findall(r'\$\$([^$]+)\$\$', content)
    for block in math_blocks:
        if re.search(r'(?<!\\)#', block):
            issues.append("Unescaped # character in math block")
    
    # Check for old LaTeX delimiters
    if re.search(r'\\\\\\\[', content):
        issues.append("Contains old LaTeX delimiters \\[")
    
    return issues

def check_code_issues(content, chapter_num):
    """Check for code formatting issues"""
    issues = []
    
    # Check for line numbers in code blocks
    code_blocks = re.findall(r'```[a-zA-Z]*\n(.*?)\n```', content, re.DOTALL)
    for i, block in enumerate(code_blocks):
        lines = block.split('\n')
        for line in lines:
            if re.match(r'^\s*\d+\s+', line):
                issues.append(f"Code block {i+1} contains line numbers")
                break
    
    # Check for standalone line numbers
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if re.match(r'^\s*\d+\s*$', line):
            issues.append(f"Standalone line number at line {i+1}")
    
    return issues

def check_code_examples_sync(chapter_num):
    """Check if code examples exist for this chapter"""
    chapter_dir = f"code_examples/chapter_{chapter_num:02d}"
    if not os.path.exists(chapter_dir):
        return ["No code examples directory"]
    
    py_files = [f for f in os.listdir(chapter_dir) if f.endswith('.py')]
    if not py_files:
        return ["No Python files in code examples directory"]
    
    return []

def verify_chapter(chapter_file):
    """Verify a single chapter"""
    chapter_num = int(re.search(r'(\d+)', chapter_file).group(1))
    
    with open(f"_chapters/{chapter_file}", 'r') as f:
        content = f.read()
    
    print(f"\n📖 Verifying Chapter {chapter_num}...")
    
    # Check math issues
    math_issues = check_math_issues(content, chapter_num)
    if math_issues:
        print(f"   ❌ Math issues found:")
        for issue in math_issues:
            print(f"      - {issue}")
    else:
        print(f"   ✅ Math rendering: OK")
    
    # Check code issues
    code_issues = check_code_issues(content, chapter_num)
    if code_issues:
        print(f"   ❌ Code issues found:")
        for issue in code_issues:
            print(f"      - {issue}")
    else:
        print(f"   ✅ Code formatting: OK")
    
    # Check code examples sync
    sync_issues = check_code_examples_sync(chapter_num)
    if sync_issues:
        print(f"   ⚠️  Code examples:")
        for issue in sync_issues:
            print(f"      - {issue}")
    else:
        print(f"   ✅ Code examples: Available")
    
    total_issues = len(math_issues) + len(code_issues) + len(sync_issues)
    return total_issues

def main():
    print("🔍 SYSTEMATIC CHAPTER VERIFICATION")
    print("=" * 50)
    
    chapter_files = sorted([f for f in os.listdir("_chapters") if f.endswith(".md")])
    
    total_issues = 0
    chapters_with_issues = 0
    
    for chapter_file in chapter_files:
        issues = verify_chapter(chapter_file)
        total_issues += issues
        if issues > 0:
            chapters_with_issues += 1
    
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print(f"Total chapters checked: {len(chapter_files)}")
    print(f"Chapters with issues: {chapters_with_issues}")
    print(f"Total issues found: {total_issues}")
    
    if total_issues == 0:
        print("🎉 ALL CHAPTERS VERIFIED - NO ISSUES FOUND!")
    else:
        print("⚠️  Issues found - see details above")
    
    # Check code examples summary
    print(f"\n📁 CODE EXAMPLES SUMMARY")
    total_py_files = 0
    chapters_with_code = 0
    
    for i in range(1, 30):
        chapter_dir = f"code_examples/chapter_{i:02d}"
        if os.path.exists(chapter_dir):
            py_files = [f for f in os.listdir(chapter_dir) if f.endswith('.py')]
            if py_files:
                total_py_files += len(py_files)
                chapters_with_code += 1
    
    print(f"Chapters with code examples: {chapters_with_code}")
    print(f"Total Python files extracted: {total_py_files}")

if __name__ == "__main__":
    main()
