#!/usr/bin/env python3
"""
Script to add Apache License 2.0 headers to all source files.

This script automatically adds the Apache License 2.0 header to all Python files
in the ROS2 Pharmaceutical IV Bag Vision System project.

Usage:
    python3 scripts/add_license_headers.py

Copyright 2025 inte-R-action Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import re
from pathlib import Path
from typing import List

# Apache License 2.0 header template
LICENSE_HEADER = '''
Copyright 2025 inte-R-action Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

def has_license_header(content: str) -> bool:
    """Check if file already has a license header."""
    return "Licensed under the Apache License" in content

def add_license_to_python_file(file_path: Path) -> bool:
    """Add license header to a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if already has license
        if has_license_header(content):
            print(f"  ✓ Already licensed: {file_path}")
            return False
        
        lines = content.split('\n')
        
        # Find the end of the existing docstring
        shebang_line = ""
        docstring_start = -1
        docstring_end = -1
        
        # Check for shebang
        if lines[0].startswith('#!'):
            shebang_line = lines[0]
            start_idx = 1
        else:
            start_idx = 0
        
        # Find docstring boundaries
        in_docstring = False
        quote_type = None
        
        for i, line in enumerate(lines[start_idx:], start_idx):
            stripped = line.strip()
            
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_start = i
                    quote_type = stripped[:3]
                    in_docstring = True
                    
                    # Check if docstring ends on same line
                    if stripped.count(quote_type) >= 2:
                        docstring_end = i
                        break
                elif stripped and not stripped.startswith('#'):
                    # No docstring found, insert before first non-comment line
                    break
            else:
                if quote_type in line:
                    docstring_end = i
                    break
        
        # Construct new content
        new_lines = []
        
        # Add shebang if present
        if shebang_line:
            new_lines.append(shebang_line)
        
        # Add existing docstring with license
        if docstring_start != -1 and docstring_end != -1:
            # Extract existing docstring content
            docstring_lines = lines[docstring_start:docstring_end + 1]
            
            # Modify the last line of docstring to add license
            if docstring_lines[-1].strip() == '"""' or docstring_lines[-1].strip() == "'''":
                # Multi-line docstring
                docstring_lines.insert(-1, LICENSE_HEADER)
            else:
                # Single-line docstring - convert to multi-line
                quote = docstring_lines[0][:3]
                docstring_content = docstring_lines[0][3:-3]
                docstring_lines = [
                    quote,
                    docstring_content,
                    LICENSE_HEADER,
                    quote
                ]
            
            new_lines.extend(docstring_lines)
            
            # Add rest of the file
            new_lines.extend(lines[docstring_end + 1:])
        else:
            # No docstring, add license as comment block
            license_comment = []
            for line in LICENSE_HEADER.strip().split('\n'):
                if line.strip():
                    license_comment.append(f"# {line}")
                else:
                    license_comment.append("#")
            
            new_lines.extend(license_comment)
            new_lines.append("")  # Empty line
            new_lines.extend(lines[start_idx:])
        
        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        
        print(f"  ✓ Added license: {file_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")
        return False

def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in the directory."""
    python_files = []
    
    # Skip certain directories
    skip_dirs = {'__pycache__', '.git', 'build', 'install', 'log', '.pytest_cache'}
    
    for root, dirs, files in os.walk(directory):
        # Remove skip directories from dirs list to avoid traversing them
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    return python_files

def main():
    """Main function to add license headers to all Python files."""
    print("Adding Apache License 2.0 headers to Python files...")
    print("=" * 60)
    
    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Find all Python files
    python_files = find_python_files(project_root)
    
    if not python_files:
        print("No Python files found.")
        return
    
    print(f"Found {len(python_files)} Python files.")
    print()
    
    # Process each file
    updated_count = 0
    for file_path in sorted(python_files):
        relative_path = file_path.relative_to(project_root)
        if add_license_to_python_file(file_path):
            updated_count += 1
    
    print()
    print("=" * 60)
    print(f"License headers added to {updated_count} files.")
    print(f"Total Python files: {len(python_files)}")
    print()
    print("Note: You should also add license headers to C++ files (.cpp, .hpp)")
    print("and other source files as needed.")

if __name__ == "__main__":
    main()