"""
setup_env.py
============
Script en Python para configurar el entorno virtual, instalar dependencias
y registrar el kernel de Jupyter para el Miniproyecto 2 (RNN - IMDb).

Uso:
    python setup_env.py
"""

import subprocess
import sys
import os
from pathlib import Path

ENV_NAME    = "env_miniproyecto2"
KERNEL_NAME = "Python (MiniProyecto 2 - RNN)"
REQ_FILE    = "requirements.txt"

def run(cmd, check=True):
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)

def main():
    base = Path(__file__).parent
    
    # Buscar requirements.txt en la carpeta Setup o en la raíz
    req_path = base / REQ_FILE
    if not req_path.exists():
        req_path = base.parent / REQ_FILE

    print("\n" + "="*60)
    print(" [1/4] Creando entorno virtual:", ENV_NAME)
    print("="*60)
    run([sys.executable, "-m", "venv", str(base / ENV_NAME)])

    if os.name == "nt":
        python_venv = str(base / ENV_NAME / "Scripts" / "python.exe")
        pip_venv    = str(base / ENV_NAME / "Scripts" / "pip.exe")
    else:
        python_venv = str(base / ENV_NAME / "bin" / "python")
        pip_venv    = str(base / ENV_NAME / "bin" / "pip")

    print("\n" + "="*60)
    print(" [2/4] Actualizando gestor pip")
    print("="*60)
    run([python_venv, "-m", "pip", "install", "--upgrade", "pip"])

    print("\n" + "="*60)
    print(f" [3/4] Instalando dependencias desde {req_path.name}")
    print("="*60)
    run([pip_venv, "install", "-r", str(req_path)])

    print("\n" + "="*60)
    print(" [4/4] Registrando kernel de Jupyter")
    print("="*60)
    run([
        python_venv, "-m", "ipykernel", "install",
        "--user",
        "--name",         ENV_NAME,
        "--display-name", KERNEL_NAME,
    ])

    print("\n" + "="*60)
    print(f" [✓] Entorno listo: {ENV_NAME}")
    print(f" [✓] Kernel registrado: \"{KERNEL_NAME}\"")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
