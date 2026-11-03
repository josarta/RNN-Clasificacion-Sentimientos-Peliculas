"""
setup_env.py
============
Script en Python para configurar el entorno virtual en la raíz del proyecto,
instalar dependencias y registrar el kernel de Jupyter para el Miniproyecto 2 (RNN - IMDb).

Uso:
    python Setup/setup_env.py
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
    # Ubicar siempre la raíz de MiniProyecto2 (carpeta padre de Setup)
    setup_dir = Path(__file__).parent
    root_dir = setup_dir.parent if setup_dir.name.lower() == "setup" else setup_dir
    
    req_path = root_dir / REQ_FILE
    if not req_path.exists():
        req_path = setup_dir / REQ_FILE

    target_env = root_dir / ENV_NAME

    print("\n" + "="*60)
    print(" [1/4] Creando entorno virtual en la raíz:", target_env)
    print("="*60)
    run([sys.executable, "-m", "venv", str(target_env)])

    if os.name == "nt":
        python_venv = str(target_env / "Scripts" / "python.exe")
        pip_venv    = str(target_env / "Scripts" / "pip.exe")
    else:
        python_venv = str(target_env / "bin" / "python")
        pip_venv    = str(target_env / "bin" / "pip")

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
    print(f" [✓] Entorno listo en: {target_env}")
    print(f" [✓] Kernel registrado: \"{KERNEL_NAME}\"")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
