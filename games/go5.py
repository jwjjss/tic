"""Simplified 5x5 Go (no ko, no pass)."""

from games.base import Game

BOARD_SIZE = 5


def _idx_to_rc(idx):
    return idx // BOARD_SIZE, idx % BOARD_SIZE


def _rc_to_idx(row, col):
    return row * BOARD_SIZE + col


def _neighbors(idx):
    row, col = _idx_to_rc(idx)
    neighbors = []
    if row > 0:
        neighbors.append(_rc_to_idx(row - 1, col))
    if row < BOARD_SIZE - 1:
        neighbors.append(_rc_to_idx(row + 1, col))
    if col > 0:
        neighbors.append(_rc_to_idx(row, col - 1))
    if col < BOARD_SIZE - 1:
        neighbors.append(_rc_to_idx(row, col + 1))
    return neighbors


def _group_and_liberties(board, start_idx):
    color = board[start_idx]
    group = set()
    liberties = set()
    stack = [start_idx]
    while stack:
        idx = stack.pop()
        if idx in group:
            continue
        group.add(idx)
        for nb in _neighbors(idx):
            if board[nb] == 0:
                liberties.add(nb)
            elif board[nb] == color and nb not in group:
                stack.append(nb)
    return group, liberties


class Go5Game(Game):
    name = "go5"
    board_size = BOARD_SIZE * BOARD_SIZE
    action_size = BOARD_SIZE * BOARD_SIZE
    max_moves = 50

    def initial_board(self):
        return [0] * self.board_size

    def legal_moves(self, board, player):
        moves = []
        for idx, value in enumerate(board):
            if value != 0:
                continue
            try:
                self.apply_move(board, idx, player)
            except ValueError:
                continue
            moves.append(idx)
        return moves

    def apply_move(self, board, move, player):
        if board[move] != 0:
            raise ValueError(f"Illegal move {move}")
        next_board = list(board)
        next_board[move] = player

        for nb in _neighbors(move):
            if next_board[nb] == -player:
                group, liberties = _group_and_liberties(next_board, nb)
                if not liberties:
                    for idx in group:
                        next_board[idx] = 0

        group, liberties = _group_and_liberties(next_board, move)
        if not liberties:
            raise ValueError(f"Suicide move {move}")
        return next_board

    def check_winner(self, board, player_to_move=None):
        if player_to_move is not None and not self.legal_moves(board, player_to_move):
            return self.max_moves_result(board, player_to_move)
        if any(value == 0 for value in board):
            return None
        score = sum(board)
        if score > 0:
            return 1
        if score < 0:
            return -1
        return 0

    def encode(self, board, player):
        return [value * player for value in board]

    def max_moves_result(self, board, player_to_move=None):
        black_libs = _liberties_for_player(board, 1)
        white_libs = _liberties_for_player(board, -1)
        if black_libs > white_libs:
            return 1
        if white_libs > black_libs:
            return -1
        return 0


def _liberties_for_player(board, player):
    visited = set()
    liberties = set()
    for idx, value in enumerate(board):
        if value == player and idx not in visited:
            group, group_libs = _group_and_liberties(board, idx)
            visited.update(group)
            liberties.update(group_libs)
    return len(liberties)
