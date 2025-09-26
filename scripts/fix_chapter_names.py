import os
import re

chapters_dir = os.path.join(os.path.dirname(__file__), '../_chapters')

if not os.path.isdir(chapters_dir):
    print(f"Error: Directory not found at {chapters_dir}")
    exit()

files = os.listdir(chapters_dir)

for f in files:
    if f.endswith('.md'):
        # Check if the filename has the pattern XX-XX-title.md
        match = re.match(r'(\d+)-\d+-(.+)\.md', f)
        if match:
            chapter_num = match.group(1)
            title = match.group(2)
            new_filename = f'{chapter_num}-{title}.md'
            
            original_path = os.path.join(chapters_dir, f)
            new_path = os.path.join(chapters_dir, new_filename)
            
            if original_path != new_path:
                os.rename(original_path, new_path)
                print(f"Renamed {f} to {new_filename}")

print("\nChapter filename fix complete.")
