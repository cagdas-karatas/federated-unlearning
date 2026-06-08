import json
import sys
import io

# Fix stdout encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Read old and current notebooks
with open('old_nb_80c486e.json', 'r', encoding='utf-8') as f:
    old_nb = json.load(f)

with open('federated_learning.ipynb', 'r', encoding='utf-8') as f:
    current_nb = json.load(f)

print("=" * 80)
print("COMPARISON: ORIGINAL (80c486e) vs CURRENT (HEAD)")
print("=" * 80)

# Cell b81d46f0 - FedEraser
print("\n\n" + "=" * 80)
print("CELL b81d46f0 - FedEraser Implementation")
print("=" * 80)

print("\n--- ORIGINAL (80c486e) ---")
found_old = False
for cell in old_nb.get('cells', []):
    if cell.get('id') == 'b81d46f0':
        found_old = True
        source_text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        print(source_text)
        break

if not found_old:
    print("NOT FOUND in old notebook")

print("\n--- CURRENT (HEAD) ---")
for cell in current_nb.get('cells', []):
    if cell.get('id') == 'b81d46f0':
        source_text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        print(source_text)
        break

# Cell 2dc3df0a - FL training with delta collection
print("\n\n" + "=" * 80)
print("CELL 2dc3df0a - Federated Learning with Delta Collection")
print("=" * 80)

print("\n--- ORIGINAL (80c486e) ---")
found_old = False
for cell in old_nb.get('cells', []):
    if cell.get('id') == '2dc3df0a':
        found_old = True
        source_text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        print(source_text)
        break

if not found_old:
    print("NOT FOUND in old notebook")

print("\n--- CURRENT (HEAD) ---")
for cell in current_nb.get('cells', []):
    if cell.get('id') == '2dc3df0a':
        source_text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        print(source_text)
        break

