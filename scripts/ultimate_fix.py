#!/usr/bin/env python3
"""
Ultimate fix for all remaining issues:
1. Fix Jekyll configuration to disable line numbers
2. Ensure proper markdown processing
3. Force complete rebuild
"""

import os
import re
from datetime import datetime

def fix_jekyll_config():
    """Fix Jekyll configuration to disable line numbers and ensure proper markdown"""
    
    with open("_config.yml", "r") as f:
        config = f.read()
    
    # Fix line numbers setting
    config = re.sub(r'line_numbers: true', 'line_numbers: false', config)
    
    # Ensure proper markdown processing
    if 'parse_block_html: true' not in config:
        # Add after kramdown section
        config = re.sub(
            r'(kramdown:.*?start_line: 1)',
            r'\1\n  parse_block_html: true\n  hard_wrap: false',
            config,
            flags=re.DOTALL
        )
    
    # Update timestamp for rebuild
    timestamp = datetime.now().isoformat()
    config = re.sub(r'# Last rebuild:.*', f'# Last rebuild: {timestamp}', config)
    
    with open("_config.yml", "w") as f:
        f.write(config)
    
    print("✅ Fixed Jekyll configuration")

def create_custom_css():
    """Create custom CSS to override any remaining line number styling"""
    
    css_content = """/* Custom CSS to ensure clean code blocks */

/* Remove line numbers from code blocks */
.highlight .lineno {
    display: none !important;
}

.highlight pre {
    counter-reset: none !important;
}

.highlight pre code {
    counter-increment: none !important;
}

.highlight pre code:before {
    content: none !important;
}

/* Ensure proper markdown rendering */
.markdown-body h1,
.markdown-body h2,
.markdown-body h3,
.markdown-body h4,
.markdown-body h5,
.markdown-body h6 {
    font-weight: 600;
    line-height: 1.25;
    margin-bottom: 16px;
    margin-top: 24px;
}

/* Clean code block styling */
.highlight {
    background: #f6f8fa;
    border-radius: 6px;
    padding: 16px;
    overflow: auto;
    margin: 16px 0;
}

.highlight pre {
    background: transparent;
    border: 0;
    font-size: 85%;
    line-height: 1.45;
    margin: 0;
    overflow: visible;
    padding: 0;
    word-wrap: normal;
}

/* Ensure math renders properly */
.MathJax {
    display: inline-block !important;
}

.MathJax_Display {
    text-align: center !important;
    margin: 1em 0 !important;
}
"""
    
    # Ensure assets/css directory exists
    os.makedirs("assets/css", exist_ok=True)
    
    with open("assets/css/custom.scss", "w") as f:
        f.write("---\n---\n\n" + css_content)
    
    print("✅ Created custom CSS to override line numbers")

def fix_layout_file():
    """Ensure the layout file properly processes markdown"""
    
    layout_content = """<!DOCTYPE html>
<html lang="{{ site.lang | default: "en-US" }}">
  <head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <title>{{ page.title | default: site.title | default: site.github.repository_name }}</title>
    <meta name="description" content="{{ page.description | default: site.description | default: site.github.project_tagline }}"/>

    <link rel="stylesheet" href="{{ "/assets/css/style.css?v=" | append: site.github.build_revision | relative_url }}">
    <link rel="stylesheet" href="{{ "/assets/css/custom.scss" | relative_url }}">
    
    <!-- MathJax Configuration -->
    <script>
      MathJax = {
        tex: {
          inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
          displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
          processEscapes: true,
          processEnvironments: true
        },
        options: {
          ignoreHtmlClass: ".*|",
          processHtmlClass: "arithmatex"
        }
      };
    </script>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    
    <!-- Syntax Highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
    <script>hljs.highlightAll();</script>
  </head>
  <body>
    <div class="container-lg px-3 my-5 markdown-body">
      
      <h1><a href="{{ "/" | absolute_url }}">{{ site.title | default: site.github.repository_name }}</a></h1>
      
      <nav>
        <a href="{{ "/" | absolute_url }}">Home</a> |
        <a href="{{ "/chapters/" | absolute_url }}">Chapters</a> |
        <a href="{{ "/code_examples/" | absolute_url }}">Code Examples</a>
      </nav>
      
      {{ content }}
      
      <footer>
        <p><small>Hosted on GitHub Pages &mdash; Theme by <a href="https://github.com/orderedlist">orderedlist</a></small></p>
      </footer>
    </div>
    <script src="{{ "/assets/js/scale.fix.js" | relative_url }}"></script>
  </body>
</html>"""
    
    # Ensure _layouts directory exists
    os.makedirs("_layouts", exist_ok=True)
    
    with open("_layouts/default.html", "w") as f:
        f.write(layout_content)
    
    print("✅ Updated layout file for proper markdown processing")

def main():
    print("🚀 ULTIMATE FIX FOR ALL REMAINING ISSUES")
    print("=" * 50)
    
    # Fix Jekyll configuration
    fix_jekyll_config()
    
    # Create custom CSS
    create_custom_css()
    
    # Fix layout file
    fix_layout_file()
    
    print("\n✅ ALL FIXES APPLIED!")
    print("\nThis should resolve:")
    print("- Line numbers in code blocks (disabled in config + CSS override)")
    print("- Markdown rendering issues (proper layout + processing)")
    print("- Math equation display (updated MathJax config)")
    print("- Code syntax highlighting (highlight.js integration)")
    
    print("\n🔄 Commit and push to deploy these fixes")

if __name__ == "__main__":
    main()
