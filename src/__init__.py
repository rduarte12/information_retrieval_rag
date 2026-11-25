"""Pacote do projeto.

Garante que os módulos de src/ também sejam importáveis pelo nome curto (ex.: o
`from inverted_index import InvertedIndex` de src/bm25.py) quando um módulo é executado
com `python -m src.<modulo>` a partir da raiz, sem precisar definir PYTHONPATH.
"""
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.append(_SRC)
