import pytest

from game_logic import apply_move, available_moves, check_winner


def test_winner_rows():
    board = [1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert check_winner(board) == 1
    board = [0, 0, 0, -1, -1, -1, 0, 0, 0]
    assert check_winner(board) == -1


def test_winner_diagonal():
    board = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert check_winner(board) == 1


def test_draw():
    board = [1, -1, 1, 1, -1, -1, -1, 1, -1]
    assert check_winner(board) == 0


def test_available_moves():
    board = [1, 0, -1, 0, 1, 0, -1, 0, 0]
    assert available_moves(board) == [1, 3, 5, 7, 8]


def test_apply_move_illegal():
    board = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        apply_move(board, 0, -1)
