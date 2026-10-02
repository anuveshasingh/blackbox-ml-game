#!/bin/sh
set -eu

if command -v uv >/dev/null 2>&1; then
    run_command="uv run --extra binary python -m PyInstaller"
else
    python_bin="${PYTHON:-python3}"
    run_command="$python_bin -m PyInstaller"
fi

$run_command \
    --noconfirm \
    --clean \
    --onefile \
    --name blackbox-ml-game \
    play.py

printf 'Built dist/blackbox-ml-game\n'
