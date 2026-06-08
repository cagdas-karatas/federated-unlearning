import json
import sys
import io

# Fix stdout encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    with open('old_nb_80c486e.json', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    keywords = ['fedraser', 'eraser', 'unlearn', 'delta_history', 'server_based', 'phase 0', 'client_based']
    
    print('=' * 80)
    print('FedEraser-related cells from commit 80c486e:')
    print('=' * 80)
    
    found_any = False
    for i, cell in enumerate(nb.get('cells', [])):
        source_text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        source_lower = source_text.lower()
        
        if any(kw in source_lower for kw in keywords):
            found_any = True
            print(f'\nCELL #{i} (ID: {cell.get("id", "unknown")})')
            print('-' * 80)
            print(source_text)
            print('-' * 80)
    
    if not found_any:
        print('\nNo FedEraser-related cells found.')
    
    print('\n' + '=' * 80)
    print('END OF SEARCH')
    print('=' * 80)
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
