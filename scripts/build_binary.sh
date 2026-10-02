#!/bin/sh
set -eu

python_bin="${PYTHON:-python3}"
"$python_bin" -m PyInstaller \
    --noconfirm \
    --clean \
    --onefile \
    --name blackbox-ml-game \
    play.py

printf 'Built dist/blackbox-ml-game\n'
