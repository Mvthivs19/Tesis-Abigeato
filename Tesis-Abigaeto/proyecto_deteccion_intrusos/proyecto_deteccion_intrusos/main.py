"""
Punto de entrada único del Sistema de Detección de Intrusos vs. Ganado.

Ejecuta: python main.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gui.app import Application


def main():
    app = Application()
    app.run()


if __name__ == "__main__":
    main()
