#!/bin/bash

# =================================================================
#  SUDOKU REASONING: MASTER PIPELINE
#  Models: Baseline, DeepThinking, FeedForward, TRM, NeuralOperator
# =================================================================

# Exit on error
set -e

echo ">>> Setting up environment..."
# Ensure packages exist so python imports work
touch experiments/__init__.py
touch experiments/baseline_transformer/__init__.py
touch experiments/deep_thinking/__init__.py
touch experiments/feed_forward_neural_net/__init__.py
touch experiments/trm/__init__.py
touch experiments/neural_operators/__init__.py

# -----------------------------------------------------------------
# 1. DATA PREPROCESSING
# -----------------------------------------------------------------
echo ""
echo "=========================================="
echo " [1/4] PROCESSING DATA"
echo "=========================================="
if [ ! -f "data/processed/train/all__inputs.npy" ]; then
    python scripts/preprocess.py
else
    echo "Data found. Skipping preprocessing."
fi

# -----------------------------------------------------------------
# 2. TRAINING
# -----------------------------------------------------------------
echo ""
echo "=========================================="
echo " [2/4] TRAINING MODELS (Scaled Configs)"
echo "=========================================="

# Comment out lines if you want to skip specific models

echo ">> 1. Baseline Transformer..."
python experiments/baseline_transformer/train.py

echo ">> 2. TRM (Tiny Recursive Model)..."
python experiments/trm/train.py

echo ">> 3. Deep Thinking 2D..."
python experiments/deep_thinking_2d/train.py

echo ">> 4. FeedForward 2D..."
python experiments/feedforward_2d/train.py

echo ">> 5. Neural Operator..."
python experiments/neural_operator/train.py

# -----------------------------------------------------------------
# 3. VISUALIZATION (GIFs)
# -----------------------------------------------------------------
echo ""
echo "=========================================="
echo " [3/4] GENERATING GIFS"
echo "=========================================="
# This uses the updated script to handle all 5 architectures
python scripts/generate_gifs.py

# -----------------------------------------------------------------
# 4. BENCHMARKING
# -----------------------------------------------------------------
echo ""
echo "=========================================="
echo " [4/4] RUNNING BENCHMARK"
echo "=========================================="
# This generates the bar chart comparison
python scripts/benchmark.py

echo ""
echo "=========================================="
echo " PIPELINE COMPLETE"
echo "=========================================="
echo "Outputs:"
echo " - GIFs: experiment_gifs/"
echo " - Chart: benchmark_comparison.png"
echo " - Data: benchmark_results.csv"