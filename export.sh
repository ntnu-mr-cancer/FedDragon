#!/usr/bin/env bash

./build.sh

tar -czvf config.tar.gz -C ./configs .
docker save skarrea/feddragonsub:latest | gzip -c > feddragonsub.tar.gz