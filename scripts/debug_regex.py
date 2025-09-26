import os
import re

chapters_dir = os.path.join(os.path.dirname(__file__), '../_chapters')
files = os.listdir(chapters_dir)
print(f"Files: {files}")

for f in files:
    match = re.match(r'(\d+)-', f)
    if match:
        print(f"Matched {f} with chapter number {match.group(1)}")
    else:
        print(f"No match for {f}")

