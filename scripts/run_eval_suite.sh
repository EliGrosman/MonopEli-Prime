#!/usr/bin/env bash
# Run the full evaluation suite.
#
# Launches 5 eval runs that compare MCTS and Hybrid agents with and without
# trading.  Each run writes JSON to results/ for later comparison via:
#
#   uv run python scripts/compare_results.py
#
# Usage:
#   bash scripts/run_eval_suite.sh            # all 5 runs (background)
#   bash scripts/run_eval_suite.sh --fg       # all 5 runs (foreground, sequential)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results"

GAMES=50
MAX_TURNS=2000
SEED=42
OPPONENTS="rule_based"

mkdir -p "$RESULTS_DIR"

FG=false
if [[ "${1:-}" == "--fg" ]]; then
    FG=true
fi

run() {
    local name="$1"
    shift
    local logfile="$RESULTS_DIR/${name}.log"

    echo "Starting: $name"
    if $FG; then
        uv run python "$@" 2>&1 | tee "$logfile"
    else
        nohup uv run python "$@" > "$logfile" 2>&1 &
        echo "  PID $! -> $logfile"
    fi
}

# Run 1: MCTS vs trading agents, 2p, 2000 turns
run mcts_2p_trades \
    "$SCRIPT_DIR/evaluate_mcts.py" \
    --opponents $OPPONENTS \
    --games $GAMES \
    --num-players 2 \
    --max-turns $MAX_TURNS \
    --seed $SEED \
    --enable-trades \
    --json "$RESULTS_DIR/mcts_2p_trades.json" \
    --quiet

# Run 2: MCTS vs trading agents, 4p, 2000 turns
run mcts_4p_trades \
    "$SCRIPT_DIR/evaluate_mcts.py" \
    --opponents $OPPONENTS \
    --games $GAMES \
    --num-players 4 \
    --max-turns $MAX_TURNS \
    --seed $SEED \
    --enable-trades \
    --json "$RESULTS_DIR/mcts_4p_trades.json" \
    --quiet

# Run 3: MCTS no-trade baseline, 2p, 2000 turns (control)
run mcts_2p_notrades \
    "$SCRIPT_DIR/evaluate_mcts.py" \
    --opponents $OPPONENTS \
    --games $GAMES \
    --num-players 2 \
    --max-turns $MAX_TURNS \
    --seed $SEED \
    --json "$RESULTS_DIR/mcts_2p_notrades.json" \
    --quiet

# Run 4: Hybrid + DeepSeek V3 (via OpenRouter) with trading, 2p, 2000 turns
run hybrid_2p_deepseek \
    "$SCRIPT_DIR/evaluate_hybrid.py" \
    --opponents $OPPONENTS \
    --games $GAMES \
    --num-players 2 \
    --max-turns $MAX_TURNS \
    --seed $SEED \
    --enable-trades \
    --llm openai --llm-model deepseek/deepseek-v3-0324 \
    --llm-base-url https://openrouter.ai/api \
    --json "$RESULTS_DIR/hybrid_2p_deepseek.json" \
    --quiet

# Run 5: Hybrid + fake LLM with trading, 2p, 2000 turns (control)
run hybrid_2p_fake \
    "$SCRIPT_DIR/evaluate_hybrid.py" \
    --opponents $OPPONENTS \
    --games $GAMES \
    --num-players 2 \
    --max-turns $MAX_TURNS \
    --seed $SEED \
    --enable-trades \
    --json "$RESULTS_DIR/hybrid_2p_fake.json" \
    --quiet

if $FG; then
    echo ""
    echo "All runs complete."
else
    echo ""
    echo "All runs launched in background. Monitor with:"
    echo "  tail -f $RESULTS_DIR/*.log"
    echo ""
    echo "When done, compare with:"
    echo "  uv run python scripts/compare_results.py"
fi
