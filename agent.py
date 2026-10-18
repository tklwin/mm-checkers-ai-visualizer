"""
Adversarial Search Agents for Myanmar Checkers (မြန်မာကျားကွက်)

This module implements three distinct AI decision strategies:
1. RandomAgent: Baseline agent making uniformly random valid moves (Easy).
2. HeuristicAgent: Greedy evaluation agent selecting moves that maximize
   immediate static board evaluation (Medium).
3. MinimaxAgent: Adversarial Search agent with Alpha-Beta pruning, heuristic evaluation,
   move ordering optimization, and search statistics tracking (Hard).
4. simulate_match: Multi-game benchmarking utility.
"""

import random
import time
from typing import Dict, List, Optional, Tuple
from checkers import (
    CheckersBoard, Move, opponent_of,
    PLAYER_WHITE, PLAYER_BLACK,
    EMPTY, MAN_WHITE, KING_WHITE, MAN_BLACK, KING_BLACK,
    WHITE_PIECES, BLACK_PIECES
)

# Heuristic Weights
MATERIAL_MAN_VALUE = 100
MATERIAL_KING_VALUE = 320
ADVANCEMENT_WEIGHT = 5
CENTER_WEIGHT = 12
BACK_ROW_DEFENSE_WEIGHT = 15
MOBILITY_WEIGHT = 3

CENTER_SQUARES = {
    (3, 2), (3, 4), (4, 3), (4, 5),
    (2, 3), (2, 5), (5, 2), (5, 4)
}


def evaluate_board(game: CheckersBoard, player: str) -> float:
    """
    Heuristic evaluation function for Myanmar Checkers board states.
    
    Evaluates:
    1. Material advantage (Men = 100, Flying Kings = 320)
    2. Positional advancement (encourages men to march toward the king row)
    3. Center dominance (control of central dark squares)
    4. Home-row defense (keeping back row pieces intact to deny enemy promotion)
    5. Mobility advantage (ratio/difference of available legal moves)
    
    Returns:
        Score relative to `player` (positive favors `player`, negative favors opponent).
    """
    if game.is_game_over():
        winner = game.get_winner()
        if winner == player:
            return 10000.0
        elif winner == opponent_of(player):
            return -10000.0
        else:
            return 0.0

    counts = game.piece_counts()
    opp = opponent_of(player)

    # 1. Material
    if player == PLAYER_WHITE:
        my_men, my_kings = counts[MAN_WHITE], counts[KING_WHITE]
        opp_men, opp_kings = counts[MAN_BLACK], counts[KING_BLACK]
    else:
        my_men, my_kings = counts[MAN_BLACK], counts[KING_BLACK]
        opp_men, opp_kings = counts[MAN_WHITE], counts[KING_WHITE]

    my_material = my_men * MATERIAL_MAN_VALUE + my_kings * MATERIAL_KING_VALUE
    opp_material = opp_men * MATERIAL_MAN_VALUE + opp_kings * MATERIAL_KING_VALUE
    material_score = my_material - opp_material

    # 2. Positional Advancement & Center Control & Back-Row Defense
    advancement_score = 0
    center_score = 0
    defense_score = 0

    for r in range(8):
        for c in range(8):
            piece = game.board[r][c]
            if piece == EMPTY:
                continue

            # Center control
            if (r, c) in CENTER_SQUARES:
                if (player == PLAYER_WHITE and piece in WHITE_PIECES) or (player == PLAYER_BLACK and piece in BLACK_PIECES):
                    center_score += CENTER_WEIGHT
                else:
                    center_score -= CENTER_WEIGHT

            # Advancement for Men
            if piece == MAN_WHITE:
                # White moves up (toward row 0)
                dist_traveled = 7 - r
                if player == PLAYER_WHITE:
                    advancement_score += dist_traveled * ADVANCEMENT_WEIGHT
                else:
                    advancement_score -= dist_traveled * ADVANCEMENT_WEIGHT
            elif piece == MAN_BLACK:
                # Black moves down (toward row 7)
                dist_traveled = r
                if player == PLAYER_BLACK:
                    advancement_score += dist_traveled * ADVANCEMENT_WEIGHT
                else:
                    advancement_score -= dist_traveled * ADVANCEMENT_WEIGHT

            # Back-row defense integrity (protecting promotion row)
            if piece == MAN_WHITE and r == 7:
                defense_score += BACK_ROW_DEFENSE_WEIGHT if player == PLAYER_WHITE else -BACK_ROW_DEFENSE_WEIGHT
            elif piece == MAN_BLACK and r == 0:
                defense_score += BACK_ROW_DEFENSE_WEIGHT if player == PLAYER_BLACK else -BACK_ROW_DEFENSE_WEIGHT

    # 3. Mobility advantage
    # If it is player's turn, game.get_legal_moves() is already available
    if game.current_player == player:
        my_moves_count = len(game.get_legal_moves(player))
        opp_moves_count = len(game.get_legal_moves(opp))
    else:
        opp_moves_count = len(game.get_legal_moves(opp))
        my_moves_count = len(game.get_legal_moves(player))

    mobility_score = (my_moves_count - opp_moves_count) * MOBILITY_WEIGHT

    total_score = (
        material_score +
        advancement_score +
        center_score +
        defense_score +
        mobility_score
    )
    return float(total_score)


class RandomAgent:
    """Baseline AI agent that chooses uniformly among available legal moves."""

    def __init__(self, player: str = PLAYER_BLACK):
        self.player = player

    @property
    def opponent(self) -> str:
        return opponent_of(self.player)

    def get_move(self, game: CheckersBoard) -> Optional[Move]:
        """Select a random legal move."""
        moves = game.get_legal_moves(self.player)
        if not moves:
            return None
        return random.choice(moves)


class HeuristicAgent:
    """
    Greedy AI agent that evaluates all immediate legal moves with the
    board evaluation heuristic and picks the highest-scoring move.
    """

    def __init__(self, player: str = PLAYER_BLACK):
        self.player = player

    @property
    def opponent(self) -> str:
        return opponent_of(self.player)

    def get_move(self, game: CheckersBoard) -> Optional[Move]:
        """Select move that maximizes immediate heuristic score."""
        moves = game.get_legal_moves(self.player)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]

        best_score = float('-inf')
        best_moves: List[Move] = []

        for move in moves:
            game.make_move(move)
            score = evaluate_board(game, self.player)
            game.undo_move()

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves)


class MinimaxAgent:
    """
    Minimax Adversarial Search agent with Alpha-Beta pruning
    and heuristic evaluation for Myanmar Checkers.
    
    Features:
    - Depth-limited search with Alpha-Beta cutoff (alpha >= beta).
    - Depth-discounted payoffs for terminal states (favors faster wins).
    - Move ordering: captures and promotions searched first to maximize pruning cutoffs.
    - Search statistics tracking (nodes_explored, candidate_scores, execution_time).
    """

    def __init__(
        self,
        player: str = PLAYER_BLACK,
        max_depth: int = 4,
        use_alpha_beta: bool = True
    ):
        self.player = player
        self.max_depth = max_depth
        self.use_alpha_beta = use_alpha_beta
        self.nodes_explored: int = 0
        self.candidate_scores: Dict[str, float] = {}
        self.execution_time_ms: float = 0.0

    @property
    def opponent(self) -> str:
        return opponent_of(self.player)

    def _order_moves(self, moves: List[Move]) -> List[Move]:
        """Sort moves so captures and promotions are evaluated first (speeds up alpha-beta cutoffs)."""
        return sorted(
            moves,
            key=lambda m: (m.capture_count, 1 if m.promoted else 0),
            reverse=True
        )

    def get_move(self, game: CheckersBoard) -> Optional[Move]:
        """
        Compute optimal move for current game state using Minimax
        with Alpha-Beta pruning.
        """
        moves = game.get_legal_moves(self.player)
        if not moves:
            return None

        self.nodes_explored = 0
        self.candidate_scores = {}
        t0 = time.perf_counter()

        ordered_moves = self._order_moves(moves)

        best_score = float('-inf')
        best_moves: List[Move] = []
        alpha = float('-inf')
        beta = float('inf')

        for move in ordered_moves:
            self.nodes_explored += 1
            game.make_move(move)

            if game.is_game_over():
                winner = game.get_winner()
                if winner == self.player:
                    score = 10000.0 - 1
                elif winner == self.opponent:
                    score = -10000.0 + 1
                else:
                    score = 0.0
            else:
                score = self._minimax(
                    game=game,
                    depth=1,
                    is_maximizing=False,
                    alpha=alpha,
                    beta=beta
                )

            game.undo_move()

            move_key = f"({move.from_pos[0]}, {move.from_pos[1]}) ➔ ({move.to_pos[0]}, {move.to_pos[1]})"
            if move.is_capture:
                move_key += f" [Capture {move.capture_count}]"
            if move.promoted:
                move_key += " [Crown 👑]"

            self.candidate_scores[move_key] = round(score, 1)

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

            if self.use_alpha_beta:
                alpha = max(alpha, score)

        self.execution_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        return random.choice(best_moves)

    def _minimax(
        self,
        game: CheckersBoard,
        depth: int,
        is_maximizing: bool,
        alpha: float,
        beta: float
    ) -> float:
        """Recursive minimax with Alpha-Beta pruning."""
        self.nodes_explored += 1

        if game.is_game_over():
            winner = game.get_winner()
            if winner == self.player:
                return 10000.0 - depth
            elif winner == self.opponent:
                return -10000.0 + depth
            else:
                return 0.0

        if depth >= self.max_depth:
            return evaluate_board(game, self.player)

        curr_turn = self.player if is_maximizing else self.opponent
        moves = game.get_legal_moves(curr_turn)

        if not moves:
            # Trapped player loses
            return (10000.0 - depth) if not is_maximizing else (-10000.0 + depth)

        ordered_moves = self._order_moves(moves)

        if is_maximizing:
            max_eval = float('-inf')
            for move in ordered_moves:
                game.make_move(move)
                eval_score = self._minimax(game, depth + 1, False, alpha, beta)
                game.undo_move()
                max_eval = max(max_eval, eval_score)
                if self.use_alpha_beta:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                game.make_move(move)
                eval_score = self._minimax(game, depth + 1, True, alpha, beta)
                game.undo_move()
                min_eval = min(min_eval, eval_score)
                if self.use_alpha_beta:
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Alpha cutoff
            return min_eval


def simulate_match(agent1, agent2, num_games: int = 10, max_moves_per_game: int = 120) -> Dict[str, any]:
    """
    Simulate a series of matches between agent1 and agent2.
    
    Args:
        agent1: First agent instance.
        agent2: Second agent instance.
        num_games: Total number of games to simulate.
        max_moves_per_game: Move limit before declaring a draw.
        
    Returns:
        Dict with game statistics and win/loss percentages.
    """
    results = {
        "agent1_wins": 0,
        "agent2_wins": 0,
        "draws": 0,
        "total_games": num_games
    }

    orig_p1 = agent1.player
    orig_p2 = agent2.player

    t0 = time.perf_counter()

    for g in range(num_games):
        # Alternate sides: even games agent1 is White, odd games agent1 is Black
        if g % 2 == 0:
            agent1.player = PLAYER_WHITE
            agent2.player = PLAYER_BLACK
        else:
            agent1.player = PLAYER_BLACK
            agent2.player = PLAYER_WHITE

        game = CheckersBoard()
        move_count = 0

        while not game.is_game_over() and move_count < max_moves_per_game:
            current_agent = agent1 if game.current_player == agent1.player else agent2
            move = current_agent.get_move(game)
            if move is None:
                break
            game.make_move(move)
            move_count += 1

        winner = game.get_winner()
        if winner == agent1.player:
            results["agent1_wins"] += 1
        elif winner == agent2.player:
            results["agent2_wins"] += 1
        else:
            results["draws"] += 1

    agent1.player = orig_p1
    agent2.player = orig_p2

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    results["elapsed_ms"] = elapsed_ms
    results["agent1_win_pct"] = round(results["agent1_wins"] / num_games * 100, 1)
    results["agent2_win_pct"] = round(results["agent2_wins"] / num_games * 100, 1)
    results["draw_pct"] = round(results["draws"] / num_games * 100, 1)

    return results

