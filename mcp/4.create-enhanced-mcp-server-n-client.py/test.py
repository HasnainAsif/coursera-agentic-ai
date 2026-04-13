from pathlib import Path

BASE_DIR = Path.cwd()
resolved = Path('server.py').resolve()
resolved.relative_to(BASE_DIR)

# resolved.read_text()

print(resolved)