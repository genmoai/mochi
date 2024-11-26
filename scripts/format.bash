#! /bin/bash
set -euxo pipefail
ruff format src demos
ruff check --fix --select I src
ruff check --fix --select I demos