"""Minimax search for tic-tac-toe."""

from functools import lru_cache

from game_logic import apply_move, available_moves, check_winner


@lru_cache(maxsize=None)
def _minimax_value(board_tuple, player):
    board = list(board_tuple)
    outcome = check_winner(board)
    if outcome is not None:
        if outcome == 0:
            return 0
        return 1 if outcome == player else -1

    best = -2
    for move in available_moves(board):
        next_board = apply_move(board, move, player)
        value = -_minimax_value(tuple(next_board), -player)
        if value > best:
            best = value
        if best == 1:
            break
    return best


def minimax_policy(board, player):
    moves = available_moves(board)
    if not moves:
        return [0.0] * 9, 0

    best_value = -2
    best_moves = []
    for move in moves:
        next_board = apply_move(board, move, player)
        value = -_minimax_value(tuple(next_board), -player)
        if value > best_value:
            best_value = value
            best_moves = [move]
        elif value == best_value:
            best_moves.append(move)

    policy = [0.0] * 9
    prob = 1.0 / len(best_moves)
    for move in best_moves:
        policy[move] = prob
    return policy, best_value
