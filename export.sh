#!/usr/bin/env bash

./build.sh

docker save joeranbosma/dragon_roberta_large_domain_specific_v2:latest | gzip -c > dragon_roberta_large_domain_specific_v2.tar.gz
