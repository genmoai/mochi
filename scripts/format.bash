#! /bin/bash
set -euxo pipefail
ruff format src
ruff check --fix --select I src
ruff check --fix --select I demos