import os
import re

def clean_python_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    cleaned_lines = []
    in_docstring = False
    docstring_char = None

    for line in lines:
        stripped = line.strip()

        if '"""' in stripped or "'''" in stripped:
            if not in_docstring:
                in_docstring = True
                docstring_char = '"""' if '"""' in stripped else "'''"
            elif docstring_char in stripped:
                in_docstring = False
            continue

        if in_docstring:
            continue

        if stripped.startswith('#'):
            continue

        line = re.sub(r'\s*#.*$', '', line)
        line = re.sub(r'[🚀🔥🎯🔥↑↓⭐🆕]', '', line)

        if line.strip() or not cleaned_lines or cleaned_lines[-1].strip():
            cleaned_lines.append(line)

    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    result = '\n'.join(cleaned_lines)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(result)

    return filepath

python_files = [
    'f:/HD-MIX_NET/train.py',
    'f:/HD-MIX_NET/evaluate.py',
    'f:/HD-MIX_NET/Models/hd_mixnet.py',
    'f:/HD-MIX_NET/Utils/losses.py',
    'f:/HD-MIX_NET/Utils/metrics.py',
    'f:/HD-MIX_NET/Utils/dataset.py',
    'f:/HD-MIX_NET/Utils/inference.py',
    'f:/HD-MIX_NET/Utils/transformers.py',
    'f:/HD-MIX_NET/Utils/layer_viz.py',
    'f:/HD-MIX_NET/test_backend.py',
    'f:/HD-MIX_NET/visualize_layers.py',
]

for pyfile in python_files:
    if os.path.exists(pyfile):
        clean_python_file(pyfile)
        print(f"✓ Cleaned: {pyfile}")
    else:
        print(f"✗ Not found: {pyfile}")

print("\nAll Python files cleaned!")
