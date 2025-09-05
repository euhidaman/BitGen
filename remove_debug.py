import re

# Files to clean
files = ['src/fiber_fusion.py', 'src/model.py']

for file_path in files:
    # Read the file with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove all debug print statements with 🔍 and DEBUG
    content = re.sub(r'print\(f"🔍.*?"\)', '# Debug removed', content)
    content = re.sub(r'print\(f"❌ DEBUG.*?"\)', '# Debug removed', content)
    
    # Write back with UTF-8 encoding
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f'✅ Removed debug statements from {file_path}')

print('🎉 All debug cleanup completed!')
