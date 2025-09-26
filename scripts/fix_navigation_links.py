#!/usr/bin/env python3
"""
Fix Navigation Links

This script fixes the navigation links throughout the site to match
the actual chapter permalinks, removing the -optimized suffixes.
"""

import os
import re
from pathlib import Path

def fix_chapters_page():
    """Fix the main chapters.md page links"""
    chapters_file = Path("chapters.md")
    
    if not chapters_file.exists():
        print("❌ chapters.md not found")
        return
    
    with open(chapters_file, 'r') as f:
        content = f.read()
    
    # Remove -optimized suffixes from all chapter links
    # Pattern: chapters/XX-chapter-name-optimized/ -> chapters/XX-chapter-name/
    content = re.sub(r'chapters/(\d+)-([^)]+)-optimized/', r'chapters/\1-\2/', content)
    
    # Fix specific naming inconsistencies
    replacements = {
        'chapters/21-ai-assisted-surgery/': 'chapters/21-ai-assisted-surgery-and-robotic-applications/',
        'chapters/22-drug-discovery-ai/': 'chapters/22-drug-discovery-and-development-with-ai/',
        'chapters/23-precision-medicine/': 'chapters/23-precision-medicine-and-personalized-healthcare/',
        'chapters/24-healthcare-operations-ai/': 'chapters/24-healthcare-operations-and-resource-optimization/',
        'chapters/25-quality-improvement-ai/': 'chapters/25-quality-improvement-and-patient-safety/',
        'chapters/26-emerging-technologies/': 'chapters/26-emerging-technologies-and-future-directions/',
        'chapters/27-case-studies-applications/': 'chapters/27-case-studies-and-real-world-applications/',
        'chapters/28-causal-inference/': 'chapters/28-causal-inference-in-healthcare-ai/',
        'chapters/29-environmental-sustainability/': 'chapters/29-environmental-sustainability-in-healthcare-ai/',
    }
    
    for old_link, new_link in replacements.items():
        content = content.replace(old_link, new_link)
    
    with open(chapters_file, 'w') as f:
        f.write(content)
    
    print("✅ Fixed chapters.md navigation links")

def fix_index_page():
    """Fix the main index.md page links"""
    index_file = Path("index.md")
    
    if not index_file.exists():
        print("❌ index.md not found")
        return
    
    with open(index_file, 'r') as f:
        content = f.read()
    
    # Remove -optimized suffixes from chapter links
    content = re.sub(r'chapters/(\d+)-([^)]+)-optimized/', r'chapters/\1-\2/', content)
    
    # Fix specific naming inconsistencies (same as above)
    replacements = {
        'chapters/21-ai-assisted-surgery/': 'chapters/21-ai-assisted-surgery-and-robotic-applications/',
        'chapters/22-drug-discovery-ai/': 'chapters/22-drug-discovery-and-development-with-ai/',
        'chapters/23-precision-medicine/': 'chapters/23-precision-medicine-and-personalized-healthcare/',
        'chapters/24-healthcare-operations-ai/': 'chapters/24-healthcare-operations-and-resource-optimization/',
        'chapters/25-quality-improvement-ai/': 'chapters/25-quality-improvement-and-patient-safety/',
        'chapters/26-emerging-technologies/': 'chapters/26-emerging-technologies-and-future-directions/',
        'chapters/27-case-studies-applications/': 'chapters/27-case-studies-and-real-world-applications/',
        'chapters/28-causal-inference/': 'chapters/28-causal-inference-in-healthcare-ai/',
        'chapters/29-environmental-sustainability/': 'chapters/29-environmental-sustainability-in-healthcare-ai/',
    }
    
    for old_link, new_link in replacements.items():
        content = content.replace(old_link, new_link)
    
    with open(index_file, 'w') as f:
        f.write(content)
    
    print("✅ Fixed index.md navigation links")

def verify_chapter_permalinks():
    """Verify that chapter permalinks match expected URLs"""
    chapters_dir = Path("_chapters")
    
    if not chapters_dir.exists():
        print("❌ _chapters directory not found")
        return
    
    print("\n📋 Chapter permalink verification:")
    
    for chapter_file in sorted(chapters_dir.glob("*.md")):
        with open(chapter_file, 'r') as f:
            content = f.read()
        
        # Extract permalink from front matter
        permalink_match = re.search(r'permalink:\s*([^\n]+)', content)
        if permalink_match:
            permalink = permalink_match.group(1).strip()
            print(f"  {chapter_file.name} -> {permalink}")
        else:
            print(f"  ❌ {chapter_file.name} -> NO PERMALINK FOUND")

def create_navigation_test_page():
    """Create a test page to verify all navigation links work"""
    test_content = '''---
layout: default
title: "Navigation Test"
---

# Navigation Link Test Page

This page tests all chapter navigation links to ensure they work correctly.

## Chapter Links Test

{% for chapter in site.chapters %}
- [{{ chapter.title }}]({{ chapter.url }}) - {{ chapter.title }}
{% endfor %}

## Manual Link Test

### Part I: Foundations
- [Chapter 1: Clinical Informatics](chapters/01-clinical-informatics/)
- [Chapter 2: Mathematical Foundations](chapters/02-mathematical-foundations/)
- [Chapter 3: Healthcare Data Engineering](chapters/03-healthcare-data-engineering/)
- [Chapter 4: Structured Machine Learning](chapters/04-structured-ml-clinical/)
- [Chapter 5: Reinforcement Learning](chapters/05-reinforcement-learning-healthcare/)
- [Chapter 6: Generative AI](chapters/06-generative-ai-healthcare/)
- [Chapter 7: AI Agents](chapters/07-ai-agents-healthcare/)

### Part II: Trustworthy AI
- [Chapter 8: Bias Detection](chapters/08-bias-detection-mitigation/)
- [Chapter 9: Interpretability](chapters/09-interpretability-explainability/)
- [Chapter 10: Robustness](chapters/10-robustness-security/)
- [Chapter 11: Regulatory Compliance](chapters/11-regulatory-compliance/)
- [Chapter 12: Clinical Validation](chapters/12-clinical-validation-frameworks/)

### Part III: Deployment
- [Chapter 13: Deployment Strategies](chapters/13-real-world-deployment-strategies/)
- [Chapter 14: Population Health](chapters/14-population-health-ai-systems/)
- [Chapter 15: Health Equity](chapters/15-health-equity-applications/)
- [Chapter 16: Medical Imaging](chapters/16-advanced-medical-imaging-ai/)
- [Chapter 17: Clinical NLP](chapters/17-clinical-nlp-at-scale/)
- [Chapter 18: Multimodal AI](chapters/18-multimodal-ai-systems/)
- [Chapter 19: Federated Learning](chapters/19-federated-learning-healthcare/)

### Part IV: Specialized Applications
- [Chapter 20: Edge Computing](chapters/20-edge-computing-healthcare/)
- [Chapter 21: AI-Assisted Surgery](chapters/21-ai-assisted-surgery-and-robotic-applications/)
- [Chapter 22: Drug Discovery](chapters/22-drug-discovery-and-development-with-ai/)
- [Chapter 23: Precision Medicine](chapters/23-precision-medicine-and-personalized-healthcare/)
- [Chapter 24: Healthcare Operations](chapters/24-healthcare-operations-and-resource-optimization/)
- [Chapter 25: Quality Improvement](chapters/25-quality-improvement-and-patient-safety/)

### Part V: Future Directions
- [Chapter 26: Emerging Technologies](chapters/26-emerging-technologies-and-future-directions/)
- [Chapter 27: Case Studies](chapters/27-case-studies-and-real-world-applications/)
- [Chapter 28: Causal Inference](chapters/28-causal-inference-in-healthcare-ai/)
- [Chapter 29: Environmental Sustainability](chapters/29-environmental-sustainability-in-healthcare-ai/)

If all links above work correctly, navigation is fixed!
'''
    
    with open('navigation_test.md', 'w') as f:
        f.write(test_content)
    
    print("✅ Created navigation test page: navigation_test.md")

def main():
    """Run all navigation fixes"""
    print("🔧 Starting Navigation Link Fixes...")
    print("=" * 50)
    
    try:
        print("\n1. Fixing chapters.md page...")
        fix_chapters_page()
        
        print("\n2. Fixing index.md page...")
        fix_index_page()
        
        print("\n3. Verifying chapter permalinks...")
        verify_chapter_permalinks()
        
        print("\n4. Creating navigation test page...")
        create_navigation_test_page()
        
        print("\n" + "=" * 50)
        print("✅ All navigation fixes completed!")
        print("\nNext steps:")
        print("1. Commit and push changes")
        print("2. Wait for GitHub Pages to rebuild")
        print("3. Test navigation at: /navigation_test")
        print("4. Verify all chapter links work correctly")
        
    except Exception as e:
        print(f"❌ Error during navigation fixes: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
