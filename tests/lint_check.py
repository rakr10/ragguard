"""Stand-in lint (ruff was not installable offline): syntax, unused imports, long lines."""
import ast, pathlib, sys

bad = 0
for path in [*pathlib.Path("src").rglob("*.py"), *pathlib.Path("tests").rglob("*.py")]:
    src = path.read_text()
    tree = ast.parse(src)
    imported = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names: imported[(a.asname or a.name).split(".")[0]] = n.lineno
        elif isinstance(n, ast.ImportFrom) and n.module != "__future__":
            for a in n.names: imported[a.asname or a.name] = n.lineno
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.value.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
    }
    exported = src.count("__all__")
    for name, line in imported.items():
        if name not in names and name not in src.replace(f"import {name}", "") and not exported:
            print(f"{path}:{line}: unused import {name}"); bad += 1
    for i, l in enumerate(src.splitlines(), 1):
        if len(l) > 100:
            print(f"{path}:{i}: line too long"); bad += 1
sys.exit(1 if bad else 0)
