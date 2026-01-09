"""Game interface for two-player, zero-sum board games."""


class Game:
    name = "base"
    board_size = 0
    action_size = 0
    max_moves = 0

    def initial_board(self):
        raise NotImplementedError

    def legal_moves(self, board, player):
        raise NotImplementedError

    def apply_move(self, board, move, player):
        raise NotImplementedError

    def check_winner(self, board, player_to_move=None):
        raise NotImplementedError

    def encode(self, board, player):
        raise NotImplementedError

    def max_moves_result(self, board, player_to_move=None):
        return 0
