import os
import re

chapters_dir = os.path.join(os.path.dirname(__file__), '../_chapters')

if not os.path.isdir(chapters_dir):
    print(f"Error: Directory not found at {chapters_dir}")
    exit()

files = os.listdir(chapters_dir)

chapters = {}
for f in files:
    match = re.match(r'(\d+)-.*', f)
    if match:
        chapter_num = int(match.group(1))
        if chapter_num not in chapters:
            chapters[chapter_num] = []
        chapters[chapter_num].append(f)

for chapter_num, file_list in sorted(chapters.items()):
    optimized_file = None
    for f in file_list:
        if 'optimized.md' in f:
            optimized_file = f
            break
    
    if not optimized_file:
        if file_list:
            optimized_file = file_list[0] # pick the first one if no optimized version

    if optimized_file:
        # Delete all other files
        for f_to_delete in file_list:
            if f_to_delete != optimized_file:
                try:
                    os.remove(os.path.join(chapters_dir, f_to_delete))
                    print(f"Deleted {f_to_delete}")
                except OSError as e:
                    print(f"Error deleting file {f_to_delete}: {e}")

        # Now, rename the remaining file
        clean_title = re.sub(r'^\\d+-', '', optimized_file)
        clean_title = clean_title.replace('.md', '')
        clean_title = clean_title.replace('-optimized', '')
        clean_title = clean_title.replace('-full-academic', '')
        clean_title = clean_title.replace('-updated', '')
        clean_title = clean_title.replace('-', ' ').title()
        new_filename = f'{chapter_num:02d}-{clean_title.replace(" ", "-").lower()}.md'
        
        original_path = os.path.join(chapters_dir, optimized_file)
        new_path = os.path.join(chapters_dir, new_filename)

        if original_path != new_path:
            try:
                os.rename(original_path, new_path)
                print(f"Renamed {optimized_file} to {new_filename}")
            except OSError as e:
                print(f"Error renaming file {optimized_file}: {e}")

    else:
        print(f"Warning: No suitable .md file found for chapter {chapter_num}")

print("\nChapter cleanup complete.")
