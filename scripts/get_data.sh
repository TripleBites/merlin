#!/usr/bin/env sh
# Download Tiny Shakespeare (~1.1 MB) into data/.
set -e
cd "$(dirname "$0")/.."
mkdir -p data
curl -sSfL -o data/tinyshakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
echo "data/tinyshakespeare.txt: $(wc -c < data/tinyshakespeare.txt) bytes"
