"""
Flask Web Application for Myanmar Checkers (မြန်မာကျားကွက်) AI Visualizer

Features:
- Full-stack interactive web playable board.
- Difficulty levels:
  * Easy (RandomAgent)
  * Medium (HeuristicAgent)
  * Hard (MinimaxAgent with Alpha-Beta pruning)
- Real-time Adversarial Search metrics (evaluation scores, explored nodes, time in ms).
- Automated Bot vs Bot simulation benchmark runner.
- Comprehensive Myanmar Checkers rule enforcement (Mandatory capture, Longest line, Flying Kings).
"""

import os
import time
from flask import Flask, render_template, request, jsonify
from checkers import (
    CheckersBoard, Move,
    PLAYER_WHITE, PLAYER_BLACK,
    EMPTY, MAN_WHITE, KING_WHITE, MAN_BLACK, KING_BLACK
)
from agent import (
    RandomAgent, HeuristicAgent, MinimaxAgent,
    evaluate_board, simulate_match
)

app = Flask(__name__)


def create_agent(agent_type: str, player: str, depth: int = 4, use_alpha_beta: bool = True):
    """Instantiate an agent given a string type ('random', 'heuristic', 'minimax')."""
    agent_type = (agent_type or "minimax").lower()
    if agent_type in ("random", "easy"):
        return RandomAgent(player=player)
    elif agent_type in ("heuristic", "medium"):
        return HeuristicAgent(player=player)
    else:
        return MinimaxAgent(player=player, max_depth=depth, use_alpha_beta=use_alpha_beta)


def format_candidate_analysis(scores: dict) -> dict:
    """Format numeric minimax scores into tactical interpretations."""
    analysis = {}
    for move_str, sc in scores.items():
        if sc >= 5000:
            label = f"+{sc:.0f} (Forced Win / Winning Advantage)"
            status = "win"
        elif sc <= -5000:
            label = f"{sc:.0f} (Forced Loss / Severe Deficit)"
            status = "loss"
        elif sc > 50:
            label = f"+{sc:.0f} (Substantial Positional Advantage)"
            status = "advantage"
        elif sc < -50:
            label = f"{sc:.0f} (Opponent Positional Advantage)"
            status = "disadvantage"
        else:
            label = f"{sc:.0f} (Balanced Position)"
            status = "even"

        analysis[move_str] = {
            "score": sc,
            "label": label,
            "status": status
        }
    return analysis


@app.route("/")
def index():
    """Render the interactive game visualizer frontend."""
    return render_template("index.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    """Initialize and return a fresh game board."""
    data = request.get_json() or {}
    starting_player = data.get("starting_player", PLAYER_WHITE)
    game = CheckersBoard(current_player=starting_player)
    return jsonify(game.to_dict())


@app.route("/api/evaluate", methods=["POST"])
def evaluate_position():
    """
    Analyze the current board state with Minimax to return
    candidate move scores and explored node count for visualization.
    """
    data = request.get_json() or {}
    board = data.get("board")
    current_player = data.get("current_player", PLAYER_WHITE)
    depth = int(data.get("depth", 4))
    use_alpha_beta = bool(data.get("use_alpha_beta", True))

    game = CheckersBoard(board=board, current_player=current_player)

    if game.is_game_over():
        return jsonify({
            "candidate_scores": {},
            "candidate_details": {},
            "nodes_explored": 0,
            "best_move": None,
            "eval_time_ms": 0.0,
            "static_eval": evaluate_board(game, current_player)
        })

    agent = MinimaxAgent(player=current_player, max_depth=depth, use_alpha_beta=use_alpha_beta)
    t0 = time.perf_counter()
    best_move = agent.get_move(game)
    eval_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    return jsonify({
        "candidate_scores": agent.candidate_scores,
        "candidate_details": format_candidate_analysis(agent.candidate_scores),
        "nodes_explored": agent.nodes_explored,
        "best_move": best_move.to_dict() if best_move else None,
        "eval_time_ms": eval_time_ms,
        "static_eval": evaluate_board(game, current_player)
    })


@app.route("/api/move", methods=["POST"])
def process_move():
    """
    Process a move: either execute a human move, an AI move, or both.
    """
    data = request.get_json() or {}
    board = data.get("board")
    current_player = data.get("current_player", PLAYER_WHITE)
    difficulty = data.get("difficulty", "hard")
    depth = int(data.get("depth", 4))
    use_alpha_beta = bool(data.get("use_alpha_beta", True))

    # Optional human move: { from: [r, c], to: [r, c] }
    player_move_data = data.get("player_move")
    is_ai_turn = bool(data.get("is_ai_turn", False))

    game = CheckersBoard(board=board, current_player=current_player)

    # 1. If human move provided, validate and execute it
    if player_move_data:
        from_pos = tuple(player_move_data.get("from"))
        to_pos = tuple(player_move_data.get("to"))

        legal_moves = game.get_legal_moves()
        matched_move = None
        for lm in legal_moves:
            if lm.from_pos == from_pos and lm.to_pos == to_pos:
                matched_move = lm
                break

        if matched_move is None:
            return jsonify({"error": "Illegal move according to Myanmar Checkers rules."}), 400

        game.make_move(matched_move)

    # 2. Check if game is over after move
    if game.is_game_over() or not is_ai_turn:
        state = game.to_dict()
        state.update({
            "ai_move": None,
            "candidate_scores": {},
            "candidate_details": {},
            "nodes_explored": 0,
            "eval_time_ms": 0.0
        })
        return jsonify(state)

    # 3. Process AI move
    agent = create_agent(
        agent_type=difficulty,
        player=game.current_player,
        depth=depth,
        use_alpha_beta=use_alpha_beta
    )

    t0 = time.perf_counter()
    ai_move = agent.get_move(game)
    eval_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    candidate_scores = {}
    nodes_explored = 0
    if isinstance(agent, MinimaxAgent):
        candidate_scores = agent.candidate_scores
        nodes_explored = agent.nodes_explored

    if ai_move is not None:
        game.make_move(ai_move)

    state = game.to_dict()
    state.update({
        "ai_move": ai_move.to_dict() if ai_move else None,
        "candidate_scores": candidate_scores,
        "candidate_details": format_candidate_analysis(candidate_scores),
        "nodes_explored": nodes_explored,
        "eval_time_ms": eval_time_ms
    })
    return jsonify(state)


@app.route("/api/simulate", methods=["POST"])
def run_simulation():
    """
    Run automated multi-game matches between two chosen agent strategies.
    """
    data = request.get_json() or {}
    agent1_type = data.get("agent1", "minimax")
    agent2_type = data.get("agent2", "random")
    num_games = int(data.get("num_games", 10))
    num_games = max(1, min(num_games, 50))  # Bounds for responsiveness
    depth = int(data.get("depth", 3))

    agent1 = create_agent(agent1_type, PLAYER_WHITE, depth=depth)
    agent2 = create_agent(agent2_type, PLAYER_BLACK, depth=depth)

    t0 = time.perf_counter()
    results = simulate_match(agent1, agent2, num_games=num_games)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    total = results["total_games"]
    results.update({
        "agent1_type": agent1_type.capitalize(),
        "agent2_type": agent2_type.capitalize(),
        "agent1_win_pct": round(results["agent1_wins"] / total * 100, 1),
        "agent2_win_pct": round(results["agent2_wins"] / total * 100, 1),
        "draw_pct": round(results["draws"] / total * 100, 1),
        "elapsed_ms": elapsed_ms
    })

    return jsonify(results)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting Myanmar Checkers AI Visualizer on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)

