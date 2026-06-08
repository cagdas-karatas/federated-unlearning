import json

# Read current notebook
with open('federated_learning.ipynb', 'r', encoding='utf-8') as f:
    current_nb = json.load(f)

# Find cells with IDs b81d46f0 and 2dc3df0a
target_ids = ['b81d46f0', '2dc3df0a']

print("=" * 80)
print("CURRENT NOTEBOOK - MATCHING CELLS")
print("=" * 80)

for target_id in target_ids:
    found = False
    for i, cell in enumerate(current_nb.get('cells', [])):
        if cell.get('id') == target_id:
            found = True
            source_text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            print(f"\nCELL {target_id} (current notebook, cell #{i})")
            print("-" * 80)
            print(source_text)
            print("-" * 80)
            break
    if not found:
        print(f"\nCell {target_id}: NOT FOUND in current notebook")

print("\n" + "=" * 80)
