#!/bin/bash

set -euo pipefail

docker run --rm -it --gpus all -p 8080:8080 gcr.io/savvy-webbing-347620/codeformer-api-vertex:latest $@
