"""5x5 chess with check rules (no castling, no en passant)."""

from games.base import Game

BOARD_SIZE = 5
SQUARES = BOARD_SIZE * BOARD_SIZE

PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6


def _idx_to_rc(idx):
    return idx // BOARD_SIZE, idx % BOARD_SIZE


def _rc_to_idx(row, col):
    return row * BOARD_SIZE + col


def _in_bounds(row, col):
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def _add_slide_moves(board, player, start_idx, deltas, moves):
    for dr, dc in deltas:
        row, col = _idx_to_rc(start_idx)
        while True:
            row += dr
            col += dc
            if not _in_bounds(row, col):
                break
            idx = _rc_to_idx(row, col)
            if board[idx] == 0:
                moves.append((start_idx, idx))
            else:
                if board[idx] * player < 0:
                    moves.append((start_idx, idx))
                break


class Chess5Game(Game):
    name = "chess5"
    board_size = SQUARES
    action_size = SQUARES * SQUARES
    max_moves = 50

    def initial_board(self):
        return [
            ROOK,
            KNIGHT,
            BISHOP,
            QUEEN,
            KING,
            PAWN,
            PAWN,
            PAWN,
            PAWN,
            PAWN,
            0,
            0,
            0,
            0,
            0,
            -PAWN,
            -PAWN,
            -PAWN,
            -PAWN,
            -PAWN,
            -ROOK,
            -KNIGHT,
            -BISHOP,
            -QUEEN,
            -KING,
        ]

    def legal_moves(self, board, player):
        moves = []
        for from_idx, to_idx in _pseudo_moves(board, player):
            next_board = _apply_move_indices(board, from_idx, to_idx, player)
            if not _is_in_check(next_board, player):
                moves.append(from_idx * SQUARES + to_idx)
        return moves

    def apply_move(self, board, move, player):
        from_idx = move // SQUARES
        to_idx = move % SQUARES
        if board[from_idx] * player <= 0:
            raise ValueError(f"Illegal move {move}")
        if board[to_idx] * player > 0:
            raise ValueError(f"Illegal capture {move}")

        next_board = _apply_move_indices(board, from_idx, to_idx, player)

        return next_board

    def check_winner(self, board, player_to_move=None):
        if KING not in board:
            return -1
        if -KING not in board:
            return 1
        if player_to_move is None:
            return None
        legal = self.legal_moves(board, player_to_move)
        if legal:
            return None
        if _is_in_check(board, player_to_move):
            return -player_to_move
        return 0

    def encode(self, board, player):
        return [value * player for value in board]

    def max_moves_result(self, board, player_to_move=None):
        return 0


def _apply_move_indices(board, from_idx, to_idx, player):
    if board[from_idx] * player <= 0:
        raise ValueError(f"Illegal move {from_idx}->{to_idx}")
    if board[to_idx] * player > 0:
        raise ValueError(f"Illegal capture {from_idx}->{to_idx}")

    next_board = list(board)
    piece = next_board[from_idx]
    next_board[from_idx] = 0
    next_board[to_idx] = piece

    row, _ = _idx_to_rc(to_idx)
    if abs(piece) == PAWN:
        if (player == 1 and row == BOARD_SIZE - 1) or (player == -1 and row == 0):
            next_board[to_idx] = QUEEN * player
    return next_board


def _pseudo_moves(board, player):
    moves = []
    for idx, piece in enumerate(board):
        if piece * player <= 0:
            continue
        kind = abs(piece)
        row, col = _idx_to_rc(idx)

        if kind == PAWN:
            direction = 1 if player == 1 else -1
            forward = row + direction
            if _in_bounds(forward, col):
                f_idx = _rc_to_idx(forward, col)
                if board[f_idx] == 0:
                    moves.append((idx, f_idx))
                for dc in (-1, 1):
                    c_col = col + dc
                    if _in_bounds(forward, c_col):
                        c_idx = _rc_to_idx(forward, c_col)
                        if board[c_idx] * player < 0:
                            moves.append((idx, c_idx))
        elif kind == KNIGHT:
            for dr, dc in (
                (2, 1),
                (1, 2),
                (-1, 2),
                (-2, 1),
                (-2, -1),
                (-1, -2),
                (1, -2),
                (2, -1),
            ):
                r2, c2 = row + dr, col + dc
                if _in_bounds(r2, c2):
                    dst = _rc_to_idx(r2, c2)
                    if board[dst] * player <= 0:
                        moves.append((idx, dst))
        elif kind == BISHOP:
            _add_slide_moves(board, player, idx, ((1, 1), (1, -1), (-1, 1), (-1, -1)), moves)
        elif kind == ROOK:
            _add_slide_moves(board, player, idx, ((1, 0), (-1, 0), (0, 1), (0, -1)), moves)
        elif kind == QUEEN:
            _add_slide_moves(
                board,
                player,
                idx,
                ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)),
                moves,
            )
        elif kind == KING:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    r2, c2 = row + dr, col + dc
                    if _in_bounds(r2, c2):
                        dst = _rc_to_idx(r2, c2)
                        if board[dst] * player <= 0:
                            moves.append((idx, dst))
    return moves


def _find_king(board, player):
    target = KING * player
    for idx, piece in enumerate(board):
        if piece == target:
            return idx
    return None


def _is_in_check(board, player):
    king_idx = _find_king(board, player)
    if king_idx is None:
        return True
    for idx, piece in enumerate(board):
        if piece * player >= 0:
            continue
        if _piece_attacks(board, idx, abs(piece), -player, king_idx):
            return True
    return False


def _piece_attacks(board, from_idx, kind, attacker, target_idx):
    fr, fc = _idx_to_rc(from_idx)
    tr, tc = _idx_to_rc(target_idx)
    dr = tr - fr
    dc = tc - fc

    if kind == PAWN:
        step = 1 if attacker == 1 else -1
        return dr == step and abs(dc) == 1
    if kind == KNIGHT:
        return (abs(dr), abs(dc)) in ((2, 1), (1, 2))
    if kind == KING:
        return max(abs(dr), abs(dc)) == 1
    if kind == BISHOP:
        if abs(dr) != abs(dc):
            return False
        return _path_clear(board, fr, fc, tr, tc)
    if kind == ROOK:
        if dr != 0 and dc != 0:
            return False
        return _path_clear(board, fr, fc, tr, tc)
    if kind == QUEEN:
        if dr == 0 or dc == 0 or abs(dr) == abs(dc):
            return _path_clear(board, fr, fc, tr, tc)
        return False
    return False


def _path_clear(board, fr, fc, tr, tc):
    dr = tr - fr
    dc = tc - fc
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
    r, c = fr + step_r, fc + step_c
    while (r, c) != (tr, tc):
        idx = _rc_to_idx(r, c)
        if board[idx] != 0:
            return False
        r += step_r
        c += step_c
    return True
