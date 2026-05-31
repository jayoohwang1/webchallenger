#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
pushd $DIR/.. > /dev/null
pushd data > /dev/null

DATA_DIR="steps tasks"
INIT_GROUP="mygroup"
TEMPLATE_FILE="template.yaml"

for data_type in $DATA_DIR; do
    template_exists=$(ls -l $data_type/* | grep $TEMPLATE_FILE | wc -l)
    if [ $template_exists -eq 0 ]; then
        echo "No template file found in $data_type"
        continue
    fi
    USERS=$(ls -d -l $data_type/* | grep ^d | awk '{ print $NF }')
    for user_dir in $USERS; do
        pushd $user_dir > /dev/null
        mkdir -p $INIT_GROUP
        pushd $INIT_GROUP > /dev/null
        cp ../../template.yaml .
        popd > /dev/null
        popd > /dev/null
    done
done
