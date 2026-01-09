"""Tic-tac-toe game rules and helpers."""

from games.tictactoe import TicTacToeGame

_GAME = TicTacToeGame()


def available_moves(board):
    return _GAME.legal_moves(board, player=1)


def apply_move(board, move, player):
    return _GAME.apply_move(board, move, player)


def check_winner(board, player_to_move=None):
    return _GAME.check_winner(board, player_to_move)


def encode_board(board, player):
    return _GAME.encode(board, player)
