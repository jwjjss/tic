"""Game registry."""

from games.chess5 import Chess5Game
from games.go5 import Go5Game
from games.tictactoe import TicTacToeGame


def get_game(name):
    name = name.lower()
    if name in ("tictactoe", "tic", "ttt"):
        return TicTacToeGame()
    if name in ("go5", "go"):
        return Go5Game()
    if name in ("chess5", "chess"):
        return Chess5Game()
    raise ValueError(f"Unknown game: {name}")
