#!/bin/bash

ALPHA=40
D=50000

for beta in 5 25 50 75; do
  for gamma in 60 110 140; do
    echo "=== Running: alpha=$ALPHA beta=$beta gamma=$gamma D=$D ==="
    python generate_mock_data/generate_mock_formal.py --D $D --alpha $ALPHA --beta $beta --gamma $gamma --no-show
  done
done
