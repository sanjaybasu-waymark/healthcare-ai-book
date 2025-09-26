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
        
        # Extract chapter number and title from filename
        match = re.match(r'(\d+)-(.+)\.md', f)
        if match:
            chapter_num = int(match.group(1))
            title_slug = match.group(2)
            
            # Convert slug to proper title
            title = title_slug.replace('-', ' ').title()
            
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Check if front matter exists
            if content.startswith('---'):
                # Find the end of front matter
                end_index = content.find('---', 3)
                if end_index != -1:
                    # Remove existing front matter
                    content = content[end_index + 3:].lstrip()
            
            # Create standardized front matter
            front_matter = f"""---
layout: default
title: "Chapter {chapter_num}: {title}"
nav_order: {chapter_num}
parent: Chapters
---

"""
            
            # Combine front matter with content
            new_content = front_matter + content
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            
            print(f"Updated front matter for {f}")

print("\nFront matter standardization complete.")
