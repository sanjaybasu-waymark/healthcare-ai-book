#!/usr/bin/env python3
"""
Force GitHub Pages rebuild by making significant changes
"""

import os
from datetime import datetime

def force_rebuild():
    """Force a complete rebuild by updating Jekyll configuration"""
    
    # Update _config.yml with a timestamp to force rebuild
    with open("_config.yml", "r") as f:
        config = f.read()
    
    # Add a timestamp comment to force rebuild
    timestamp = datetime.now().isoformat()
    
    if "# Last rebuild:" in config:
        # Replace existing timestamp
        import re
        config = re.sub(r'# Last rebuild:.*\n', f'# Last rebuild: {timestamp}\n', config)
    else:
        # Add timestamp at the end
        config += f"\n# Last rebuild: {timestamp}\n"
    
    with open("_config.yml", "w") as f:
        f.write(config)
    
    # Also update the main index.md to trigger a change
    with open("index.md", "r") as f:
        index_content = f.read()
    
    # Add a hidden comment to force rebuild
    if "<!-- Force rebuild" in index_content:
        import re
        index_content = re.sub(r'<!-- Force rebuild.*-->', f'<!-- Force rebuild {timestamp} -->', index_content)
    else:
        index_content += f"\n<!-- Force rebuild {timestamp} -->\n"
    
    with open("index.md", "w") as f:
        f.write(index_content)
    
    print(f"✅ Added rebuild timestamp: {timestamp}")

def main():
    print("🔄 FORCING GITHUB PAGES REBUILD")
    print("=" * 40)
    
    force_rebuild()
    
    print("\n✅ Rebuild triggered!")
    print("This should force GitHub Pages to:")
    print("- Clear all caches")
    print("- Rebuild from latest source")
    print("- Deploy fresh content")
    print("\nWait 2-3 minutes for changes to appear.")

if __name__ == "__main__":
    main()
