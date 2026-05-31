#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
pushd $DIR/.. > /dev/null
source .env

# clean up venv
rm -rf venv
