"""Tic-tac-toe game implementation."""

from games.base import Game

WIN_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


class TicTacToeGame(Game):
    name = "tictactoe"
    board_size = 9
    action_size = 9
    max_moves = 9

    def initial_board(self):
        return [0] * 9

    def legal_moves(self, board, player):
        return [idx for idx, value in enumerate(board) if value == 0]

    def apply_move(self, board, move, player):
        if board[move] != 0:
            raise ValueError(f"Illegal move {move}")
        next_board = list(board)
        next_board[move] = player
        return next_board

    def check_winner(self, board, player_to_move=None):
        for a, b, c in WIN_LINES:
            total = board[a] + board[b] + board[c]
            if total == 3:
                return 1
            if total == -3:
                return -1
        if all(value != 0 for value in board):
            return 0
        return None

    def encode(self, board, player):
        return [value * player for value in board]
