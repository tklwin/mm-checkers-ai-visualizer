"""
Myanmar Checkers (မြန်မာကျားကွက်) Game Engine

Rules of Traditional Myanmar Checkers:
1. Played on an 8x8 board on the 32 dark squares ((row + col) % 2 == 1).
2. Each player starts with 12 men on their first 3 rows:
   - Black ('B') on rows 0, 1, 2 (moves downward, advancing row +1)
   - White ('W') on rows 5, 6, 7 (moves upward, advancing row -1)
3. Regular Men:
   - Move forward diagonally 1 square into an empty dark square.
   - Capture forward diagonally by jumping over an adjacent opponent piece into an empty square.
   - Regular men CANNOT capture backwards.
   - Reaching the furthest row promotes a man to a Flying King ('WK' or 'BK').
   - If a man reaches the promotion rank during a jump sequence, it is crowned immediately
     and its turn ends (it cannot continue capturing as a king on that turn).
4. Flying Kings (ကျားကြီး / ဒမ်):
   - Move diagonally in all 4 directions across any number of unoccupied squares (like a chess bishop).
   - Capture diagonally across distance: jumps over an opponent piece along a diagonal ray
     and lands on any empty square beyond it along the same ray.
   - Multi-jumps: After landing, if another capture is available along an intersecting diagonal,
     the king continues capturing.
5. Mandatory Capture & Maximum Capture (Longest Line) Rule:
   - Capturing is compulsory. If any capture is available, non-capturing moves are illegal.
   - If multiple capture sequences exist, the player MUST choose the sequence that captures
     the maximum number of pieces (အများဆုံးစားရမည့် စည်းမျဉ်း).
6. Winning & Terminal Conditions:
   - A player wins when the opponent has no remaining pieces or has no legal moves (stalemate/block).
   - Draw conditions: 40 half-moves without capture or man advancement, or 3-fold board repetition.
"""

from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
import copy

# Player definitions
PLAYER_WHITE = 'W'
PLAYER_BLACK = 'B'

# Piece definitions
EMPTY = '.'
MAN_WHITE = 'W'
KING_WHITE = 'WK'
MAN_BLACK = 'B'
KING_BLACK = 'BK'

WHITE_PIECES = {MAN_WHITE, KING_WHITE}
BLACK_PIECES = {MAN_BLACK, KING_BLACK}

# Directions for diagonals: (d_row, d_col)
UP_LEFT = (-1, -1)
UP_RIGHT = (-1, 1)
DOWN_LEFT = (1, -1)
DOWN_RIGHT = (1, 1)

ALL_DIRECTIONS = [UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT]
WHITE_FORWARD_DIRECTIONS = [UP_LEFT, UP_RIGHT]
BLACK_FORWARD_DIRECTIONS = [DOWN_LEFT, DOWN_RIGHT]


def is_playable_square(row: int, col: int) -> bool:
    """Check if (row, col) is on the 8x8 board and is a playable dark square."""
    return 0 <= row < 8 and 0 <= col < 8 and ((row + col) % 2 == 1)


def opponent_of(player: str) -> str:
    """Return the opponent player symbol ('W' <-> 'B')."""
    return PLAYER_BLACK if player == PLAYER_WHITE else PLAYER_WHITE


def get_player_pieces(player: str) -> Set[str]:
    """Return the set of piece symbols belonging to the player."""
    return WHITE_PIECES if player == PLAYER_WHITE else BLACK_PIECES


def get_opponent_pieces(player: str) -> Set[str]:
    """Return the set of piece symbols belonging to the opponent."""
    return BLACK_PIECES if player == PLAYER_WHITE else WHITE_PIECES


@dataclass
class Move:
    """
    Represents a single legal move or a multi-jump capture sequence.
    
    Attributes:
        from_pos: Starting (row, col).
        to_pos: Destination (row, col).
        path: Ordered list of intermediate square coordinates: [from_pos, hop1, hop2, ..., to_pos].
        captures: Ordered list of captured enemy piece coordinates: [cap1, cap2, ...].
        promoted: Whether the moving piece was promoted to King at the end of this move.
    """
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    path: List[Tuple[int, int]] = field(default_factory=list)
    captures: List[Tuple[int, int]] = field(default_factory=list)
    promoted: bool = False

    @property
    def is_capture(self) -> bool:
        return len(self.captures) > 0

    @property
    def capture_count(self) -> int:
        return len(self.captures)

    def to_dict(self) -> dict:
        return {
            "from": list(self.from_pos),
            "to": list(self.to_pos),
            "path": [list(p) for p in self.path],
            "captures": [list(c) for c in self.captures],
            "promoted": self.promoted,
            "is_capture": self.is_capture,
            "capture_count": self.capture_count
        }


@dataclass
class UndoRecord:
    """Data stored to backtrack a move without copying the entire board."""
    move: Move
    moving_piece: str
    captured_pieces: List[Tuple[Tuple[int, int], str]]
    prev_current_player: str
    prev_halfmove_clock: int


class CheckersBoard:
    """
    Game state representation for traditional Myanmar Checkers.
    """

    def __init__(self, board: Optional[List[List[str]]] = None, current_player: str = PLAYER_WHITE):
        if board is not None:
            self.board = [list(row) for row in board]
        else:
            self.board = self._create_initial_board()

        self.current_player = current_player
        self.move_history: List[UndoRecord] = []
        self.halfmove_clock: int = 0  # Number of half-moves since last capture or man move
        self.position_history: Dict[str, int] = {}
        self._record_position()

    def _create_initial_board(self) -> List[List[str]]:
        """Set up standard 8x8 board with 12 White and 12 Black pieces."""
        grid = [[EMPTY for _ in range(8)] for _ in range(8)]
        # Black occupies top rows: 0, 1, 2
        for r in range(3):
            for c in range(8):
                if (r + c) % 2 == 1:
                    grid[r][c] = MAN_BLACK
        # White occupies bottom rows: 5, 6, 7
        for r in range(5, 8):
            for c in range(8):
                if (r + c) % 2 == 1:
                    grid[r][c] = MAN_WHITE
        return grid

    def _record_position(self) -> None:
        key = self.get_state_hash()
        self.position_history[key] = self.position_history.get(key, 0) + 1

    def _unrecord_position(self) -> None:
        key = self.get_state_hash()
        if key in self.position_history:
            self.position_history[key] -= 1
            if self.position_history[key] <= 0:
                del self.position_history[key]

    def get_state_hash(self) -> str:
        """Create a compact string representation of the board state and turn."""
        flat = "".join("".join(row) for row in self.board)
        return f"{flat}:{self.current_player}"

    def get_piece(self, row: int, col: int) -> str:
        if 0 <= row < 8 and 0 <= col < 8:
            return self.board[row][col]
        return EMPTY

    def set_piece(self, row: int, col: int, piece: str) -> None:
        self.board[row][col] = piece

    def clone(self) -> "CheckersBoard":
        """Return a fast deep copy of this game state."""
        new_game = CheckersBoard.__new__(CheckersBoard)
        new_game.board = [list(row) for row in self.board]
        new_game.current_player = self.current_player
        new_game.move_history = []
        new_game.halfmove_clock = self.halfmove_clock
        new_game.position_history = dict(self.position_history)
        return new_game

    # -------------------------------------------------------------------------
    # Legal Move Generation with Myanmar Checkers Rules
    # -------------------------------------------------------------------------

    def get_legal_moves(self, player: Optional[str] = None) -> List[Move]:
        """
        Generate all legal moves for `player` (defaults to current_player).
        
        Enforces Myanmar rules:
        1. Captures are compulsory.
        2. Maximum capture rule: Must choose moves capturing the maximum number of pieces.
        3. Regular men capture forward only.
        4. Flying kings capture over long diagonals.
        5. If a man reaches promotion during a jump sequence, it crowns and turn ends.
        """
        p = player if player is not None else self.current_player
        all_captures: List[Move] = []
        all_steps: List[Move] = []

        my_pieces = get_player_pieces(p)

        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece in my_pieces:
                    # Collect jump sequences for this piece
                    piece_captures = self._get_captures_for_piece(r, c, piece, p)
                    all_captures.extend(piece_captures)

                    # If no captures have been found across the board yet, collect regular steps
                    # (Captures will override steps anyway)
                    piece_steps = self._get_steps_for_piece(r, c, piece, p)
                    all_steps.extend(piece_steps)

        # Mandatory Capture Rule: If any captures exist, non-capturing steps are illegal.
        if all_captures:
            max_cap = max(m.capture_count for m in all_captures)
            # Maximum Capture (Longest Line) Rule: Must take moves with the highest capture count
            best_captures = [m for m in all_captures if m.capture_count == max_cap]
            return best_captures

        return all_steps

    def _get_steps_for_piece(self, r: int, c: int, piece: str, player: str) -> List[Move]:
        """Generate regular 1-square non-capturing moves for a piece."""
        steps: List[Move] = []

        if piece in (MAN_WHITE, MAN_BLACK):
            dirs = WHITE_FORWARD_DIRECTIONS if player == PLAYER_WHITE else BLACK_FORWARD_DIRECTIONS
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if is_playable_square(nr, nc) and self.board[nr][nc] == EMPTY:
                    promoted = (player == PLAYER_WHITE and nr == 0) or (player == PLAYER_BLACK and nr == 7)
                    steps.append(Move(
                        from_pos=(r, c),
                        to_pos=(nr, nc),
                        path=[(r, c), (nr, nc)],
                        captures=[],
                        promoted=promoted
                    ))
        else:
            # Flying King: slide any distance along all 4 diagonal rays
            for dr, dc in ALL_DIRECTIONS:
                dist = 1
                while True:
                    nr, nc = r + dr * dist, c + dc * dist
                    if not is_playable_square(nr, nc):
                        break
                    if self.board[nr][nc] != EMPTY:
                        break  # Blocked by another piece
                    steps.append(Move(
                        from_pos=(r, c),
                        to_pos=(nr, nc),
                        path=[(r, c), (nr, nc)],
                        captures=[],
                        promoted=False
                    ))
                    dist += 1

        return steps

    def _get_captures_for_piece(self, r: int, c: int, piece: str, player: str) -> List[Move]:
        """Find all multi-jump capture paths originating from (r, c)."""
        paths: List[Move] = []
        is_king = (piece in (KING_WHITE, KING_BLACK))
        captured_set: Set[Tuple[int, int]] = set()

        self._find_capture_sequences(
            r=r,
            c=c,
            curr_r=r,
            curr_c=c,
            piece=piece,
            player=player,
            is_king=is_king,
            current_path=[(r, c)],
            current_captures=[],
            captured_set=captured_set,
            results=paths
        )
        return paths

    def _find_capture_sequences(
        self,
        r: int,
        c: int,
        curr_r: int,
        curr_c: int,
        piece: str,
        player: str,
        is_king: bool,
        current_path: List[Tuple[int, int]],
        current_captures: List[Tuple[int, int]],
        captured_set: Set[Tuple[int, int]],
        results: List[Move]
    ) -> None:
        """
        Recursive depth-first search for all maximal multi-jump sequences.
        """
        enemy_pieces = get_opponent_pieces(player)
        found_further_jump = False

        if not is_king:
            # Regular Man: Jumps diagonally forward only
            dirs = WHITE_FORWARD_DIRECTIONS if player == PLAYER_WHITE else BLACK_FORWARD_DIRECTIONS
            for dr, dc in dirs:
                mid_r, mid_c = curr_r + dr, curr_c + dc
                land_r, land_c = curr_r + 2 * dr, curr_c + 2 * dc

                if is_playable_square(land_r, land_c):
                    mid_piece = self.board[mid_r][mid_c]
                    land_piece = self.board[land_r][land_c]

                    # Enemy piece must be present and not yet captured in this multi-jump
                    if (mid_piece in enemy_pieces and 
                        (mid_r, mid_c) not in captured_set and 
                        (land_piece == EMPTY or (land_r, land_c) == (r, c))):

                        found_further_jump = True
                        promoted = (player == PLAYER_WHITE and land_r == 0) or (player == PLAYER_BLACK and land_r == 7)

                        captured_set.add((mid_r, mid_c))
                        next_path = current_path + [(land_r, land_c)]
                        next_captures = current_captures + [(mid_r, mid_c)]

                        if promoted:
                            # Myanmar rule: Reaching promotion rank ends the turn immediately
                            results.append(Move(
                                from_pos=(r, c),
                                to_pos=(land_r, land_c),
                                path=next_path,
                                captures=next_captures,
                                promoted=True
                            ))
                        else:
                            # Continue searching for subsequent jumps from land_r, land_c
                            self._find_capture_sequences(
                                r=r,
                                c=c,
                                curr_r=land_r,
                                curr_c=land_c,
                                piece=piece,
                                player=player,
                                is_king=False,
                                current_path=next_path,
                                current_captures=next_captures,
                                captured_set=captured_set,
                                results=results
                            )
                        captured_set.remove((mid_r, mid_c))
        else:
            # Flying King: Slides any distance along diagonal, jumps enemy, lands on any empty square beyond
            for dr, dc in ALL_DIRECTIONS:
                # 1. Slide along ray to look for the first piece
                dist = 1
                found_enemy = None
                enemy_pos = None

                while True:
                    check_r, check_c = curr_r + dr * dist, curr_c + dc * dist
                    if not is_playable_square(check_r, check_c):
                        break

                    square_piece = self.board[check_r][check_c]
                    if (check_r, check_c) in captured_set:
                        # Cannot jump over an already captured piece
                        break

                    if square_piece != EMPTY and (check_r, check_c) != (r, c):
                        if square_piece in enemy_pieces:
                            found_enemy = square_piece
                            enemy_pos = (check_r, check_c)
                        # Hit a piece (friend or foe); cannot slide further before jumping
                        break
                    dist += 1

                # If an enemy was found along this ray, inspect landing squares beyond it
                if found_enemy and enemy_pos:
                    land_dist = dist + 1
                    while True:
                        land_r, land_c = curr_r + dr * land_dist, curr_c + dc * land_dist
                        if not is_playable_square(land_r, land_c):
                            break
                        land_piece = self.board[land_r][land_c]
                        if land_piece != EMPTY and (land_r, land_c) != (r, c):
                            # Blocked by another piece
                            break

                        found_further_jump = True
                        captured_set.add(enemy_pos)
                        next_path = current_path + [(land_r, land_c)]
                        next_captures = current_captures + [enemy_pos]

                        # Continue searching for multi-jumps from this landing square
                        self._find_capture_sequences(
                            r=r,
                            c=c,
                            curr_r=land_r,
                            curr_c=land_c,
                            piece=piece,
                            player=player,
                            is_king=True,
                            current_path=next_path,
                            current_captures=next_captures,
                            captured_set=captured_set,
                            results=results
                        )
                        captured_set.remove(enemy_pos)
                        land_dist += 1

        # If this path captured at least one piece and cannot continue jumping, record terminal move
        if not found_further_jump and current_captures:
            results.append(Move(
                from_pos=(r, c),
                to_pos=(curr_r, curr_c),
                path=list(current_path),
                captures=list(current_captures),
                promoted=False
            ))

    # -------------------------------------------------------------------------
    # State Transition: Make and Undo Move
    # -------------------------------------------------------------------------

    def make_move(self, move: Move) -> bool:
        """
        Execute `move` on the board.
        Updates board state, current_player, and clocks.
        Saves undo record for fast backtracking in Minimax.
        """
        from_r, from_c = move.from_pos
        to_r, to_c = move.to_pos
        piece = self.board[from_r][from_c]

        if piece == EMPTY:
            return False

        # Prepare undo record
        captured_data: List[Tuple[Tuple[int, int], str]] = []
        for cr, cc in move.captures:
            captured_data.append(((cr, cc), self.board[cr][cc]))
            self.board[cr][cc] = EMPTY

        # Move the piece
        self.board[from_r][from_c] = EMPTY

        new_piece = piece
        if move.promoted:
            new_piece = KING_WHITE if self.current_player == PLAYER_WHITE else KING_BLACK
        elif piece in (MAN_WHITE, MAN_BLACK):
            # Check promotion upon regular step or landing
            if self.current_player == PLAYER_WHITE and to_r == 0:
                new_piece = KING_WHITE
                move.promoted = True
            elif self.current_player == PLAYER_BLACK and to_r == 7:
                new_piece = KING_BLACK
                move.promoted = True

        self.board[to_r][to_c] = new_piece

        # Record undo
        record = UndoRecord(
            move=move,
            moving_piece=piece,
            captured_pieces=captured_data,
            prev_current_player=self.current_player,
            prev_halfmove_clock=self.halfmove_clock
        )
        self.move_history.append(record)

        # Update halfmove clock (resets on capture or man move)
        if move.is_capture or piece in (MAN_WHITE, MAN_BLACK):
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # Switch player
        self.current_player = opponent_of(self.current_player)
        self._record_position()
        return True

    def undo_move(self) -> bool:
        """
        Undo the most recently executed move.
        Restores captured pieces, piece locations, clocks, and turns.
        """
        if not self.move_history:
            return False

        self._unrecord_position()
        record = self.move_history.pop()
        move = record.move

        from_r, from_c = move.from_pos
        to_r, to_c = move.to_pos

        # Remove piece from destination and restore original piece at source
        self.board[to_r][to_c] = EMPTY
        self.board[from_r][from_c] = record.moving_piece

        # Restore all captured pieces
        for (cr, cc), captured_piece in record.captured_pieces:
            self.board[cr][cc] = captured_piece

        # Restore player and clocks
        self.current_player = record.prev_current_player
        self.halfmove_clock = record.prev_halfmove_clock
        return True

    # -------------------------------------------------------------------------
    # Game Evaluation & Terminal Conditions
    # -------------------------------------------------------------------------

    def piece_counts(self) -> Dict[str, int]:
        """Count pieces currently on board."""
        counts = {
            MAN_WHITE: 0,
            KING_WHITE: 0,
            MAN_BLACK: 0,
            KING_BLACK: 0
        }
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if p in counts:
                    counts[p] += 1
        return counts

    def is_game_over(self) -> bool:
        """Check if terminal condition is reached (win, loss, or draw)."""
        counts = self.piece_counts()
        white_total = counts[MAN_WHITE] + counts[KING_WHITE]
        black_total = counts[MAN_BLACK] + counts[KING_BLACK]

        # 1. Elimination
        if white_total == 0 or black_total == 0:
            return True

        # 2. Stalemate (no legal moves for current player)
        if len(self.get_legal_moves()) == 0:
            return True

        # 3. 40 half-moves rule (no captures or man advancements)
        if self.halfmove_clock >= 40:
            return True

        # 4. Threefold repetition
        curr_key = self.get_state_hash()
        if self.position_history.get(curr_key, 0) >= 3:
            return True

        return False

    def get_winner(self) -> Optional[str]:
        """
        Determine the winner if the game has ended.
        Returns 'W', 'B', 'DRAW', or None (if ongoing).
        """
        counts = self.piece_counts()
        white_total = counts[MAN_WHITE] + counts[KING_WHITE]
        black_total = counts[MAN_BLACK] + counts[KING_BLACK]

        if white_total == 0:
            return PLAYER_BLACK
        if black_total == 0:
            return PLAYER_WHITE

        # Check legal moves for current player
        moves = self.get_legal_moves()
        if not moves:
            # Current player is trapped/blocked -> opponent wins
            return opponent_of(self.current_player)

        # Draw checks
        if self.halfmove_clock >= 40:
            return "DRAW"
        curr_key = self.get_state_hash()
        if self.position_history.get(curr_key, 0) >= 3:
            return "DRAW"

        return None

    def to_dict(self) -> dict:
        """Serialize complete game state for JSON API."""
        counts = self.piece_counts()
        legal_moves = self.get_legal_moves()
        winner = self.get_winner()
        is_over = self.is_game_over()

        return {
            "board": [list(row) for row in self.board],
            "current_player": self.current_player,
            "white_count": counts[MAN_WHITE] + counts[KING_WHITE],
            "black_count": counts[MAN_BLACK] + counts[KING_BLACK],
            "white_kings": counts[KING_WHITE],
            "black_kings": counts[KING_BLACK],
            "is_game_over": is_over,
            "winner": winner,
            "legal_moves": [m.to_dict() for m in legal_moves],
            "has_captures": any(m.is_capture for m in legal_moves),
            "halfmove_clock": self.halfmove_clock
        }

