#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
pushd $DIR/.. > /dev/null
source .env

# make sure benchmarks are setup
# git submodule update --init --recursive

# generate test data for visualwebarena
pushd webchallenger/benchmarks/visualwebarena > /dev/null
python -m venv venv
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m playwright install
venv/bin/python -m pip install -e .
venv/bin/python scripts/generate_test_data.py
mkdir -p ./.auth
venv/bin/python browser_env/auto_login.py
