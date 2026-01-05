import numpy as np
from typing import List, Optional, Tuple


class TicTacToe:
    """Simple Tic Tac Toe environment.

    Board uses 1 for player X, -1 for player O, and 0 for empty.
    The environment does not enforce which player the agent controls; callers pass
    the player marker when requesting legal moves or applying moves.
    """

    def __init__(self):
        self.board = np.zeros(9, dtype=np.int8)

    @staticmethod
    def opponent(player: int) -> int:
        return -player

    def reset(self):
        self.board[:] = 0

    def legal_actions(self) -> List[int]:
        return [i for i, v in enumerate(self.board) if v == 0]

    def apply_action(self, action: int, player: int):
        if self.board[action] != 0:
            raise ValueError("Illegal move")
        self.board[action] = player

    def check_winner(self) -> Optional[int]:
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
            (0, 4, 8), (2, 4, 6),  # diagonals
        ]
        for a, b, c in lines:
            s = self.board[a] + self.board[b] + self.board[c]
            if s == 3:
                return 1
            if s == -3:
                return -1
        if not self.legal_actions():
            return 0
        return None

    def observation(self, current_player: int) -> np.ndarray:
        """Return observation from current player's perspective."""
        return (self.board * current_player).astype(np.float32)

    def clone(self) -> "TicTacToe":
        clone_env = TicTacToe()
        clone_env.board = self.board.copy()
        return clone_env

    def __str__(self) -> str:
        symbols = {1: "X", -1: "O", 0: " "}
        rows = ["|".join(symbols[self.board[r * 3 + c]] for c in range(3)) for r in range(3)]
        return "\n-+-+-\n".join(rows)
