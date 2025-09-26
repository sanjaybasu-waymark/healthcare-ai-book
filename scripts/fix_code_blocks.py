import os
import re

chapters_dir = os.path.join(os.path.dirname(__file__), '../_chapters')

if not os.path.isdir(chapters_dir):
    print(f"Error: Directory not found at {chapters_dir}")
    exit()

files = os.listdir(chapters_dir)

for f in files:
    if f.endswith('.md'):
        file_path = os.path.join(chapters_dir, f)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Fix code blocks that might not be properly formatted
        # Ensure proper code block formatting
        # Look for patterns like "```python" or "```" and ensure they're on their own lines
        content = re.sub(r'```(\\w+)?\\n?', r'```\\1\\n', content)
        content = re.sub(r'\\n```\\n?', r'\\n```\\n', content)
        
        # Fix math equations - ensure they use proper delimiters
        # Replace \\( \\) with $ $ for inline math
        content = re.sub(r'\\\\\\\\\\((.+?)\\\\\\\\\\)', r'$\\1$', content)
        # Replace \\[ \\] with $$ $$ for display math  
        content = re.sub(r'\\\\\\\\\\[(.+?)\\\\\\\\\\]', r'$$\\1$$', content)
        
        # Ensure proper spacing around math equations
        content = re.sub(r'([^$])\\$\\$([^$])', r'\\1\\n$$\\2', content)
        content = re.sub(r'([^$])\\$\\$\\n', r'\\1\\n$$\\n', content)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        print(f"Fixed code blocks and math in {f}")

print("\\nCode block and math formatting fix complete.")
