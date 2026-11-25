"""Dónde vive cada cosa. La configuración del usuario va en la carpeta estándar de macOS, no junto al
programa (así la app puede estar en /Aplicaciones, que es de solo lectura)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Recursos (interfaz, íconos, ayudantes): junto al código, o dentro del .app empaquetado.
if getattr(sys, "frozen", False):
    RES = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
else:
    RES = Path(__file__).resolve().parent.parent

UI = RES / "ui"
BIN = RES / "bin"

DATA = Path.home() / "Library" / "Application Support" / "Surround BT"
DATA.mkdir(parents=True, exist_ok=True)
CONFIG = DATA / "config.json"
PROFILES = DATA / "profiles"

# Migración: si había configuración junto al código (versión de desarrollo), se trae una sola vez.
_old = RES / "config.json"
if not CONFIG.exists() and _old.exists():
    shutil.copy(_old, CONFIG)
    if (RES / "profiles").exists() and not PROFILES.exists():
        shutil.copytree(RES / "profiles", PROFILES)
