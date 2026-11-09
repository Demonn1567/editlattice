#!/usr/bin/env bash
# Download the three benchmarks from their official sources and check them against the SHA-256 of the
# exact files used in the paper. Usage: bash scripts/fetch_datasets.sh
set -euo pipefail
cd "$(dirname "$0")/.."
check() { echo "$2  $1" | shasum -a 256 -c - ; }
mkdir -p datasets/gsmhard data/lcb_raw data/bird
curl -L -o datasets/gsmhard/gsmhardv2.jsonl https://huggingface.co/datasets/reasoning-machines/gsm-hard/resolve/main/gsmhardv2.jsonl
check datasets/gsmhard/gsmhardv2.jsonl 2032b0188a49ac0ec38cbe1e695e7d62d6dddc44bbb64f920728aca710e9ddef
for f in test5 test6; do
  curl -L -o data/lcb_raw/$f.jsonl https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/$f.jsonl
done
check data/lcb_raw/test5.jsonl 7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214
check data/lcb_raw/test6.jsonl bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5
curl -L -o data/bird/minidev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip
(cd data/bird && unzip -q -o minidev.zip && rm minidev.zip)
check data/bird/minidev/MINIDEV/mini_dev_sqlite.json 4ba5fa8de55856222f484d380d2ba872b380bf79d825de70478e2120cb0fc43b
echo "all datasets present and verified"
