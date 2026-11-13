"""Configuração do pytest: torna os módulos de src/ e tests/reference/ importáveis."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "tests" / "reference"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
