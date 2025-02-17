#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t joeranbosma/dragon_roberta_large_domain_specific_v2:latest -t joeranbosma/dragon_roberta_large_domain_specific_v2:v0.2.1 "$SCRIPTPATH"
