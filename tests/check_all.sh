#!/usr/bin/env bash

reset

export PYTHONPATH=$PYTHONPATH:"src/"

echo "ruff:"
ruff check src/evolusnake/
ruff check tests/
ruff check examples/

echo -e "\n\n-------------------------------------------\n\n"

echo "mypy:"
mypy --check-untyped-defs src/evolusnake/
mypy --check-untyped-defs tests/
mypy --check-untyped-defs examples/

echo -e "\n\n-------------------------------------------\n\n"

echo "flake8:"
flake8 src/evolusnake/
flake8 tests/
flake8 examples/

echo -e "\n\n-------------------------------------------\n\n"

echo "pyright:"
pyright src/evolusnake/
pyright tests/
pyright examples/
