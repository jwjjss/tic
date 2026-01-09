"""Evaluate pure MCTS (random rollout) vs a random bot for multiple games."""

import argparse
import math
import random

from games import get_game


class Node:
    def __init__(self, prior):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _dirichlet(alpha, size):
    samples = [random.gammavariate(alpha, 1.0) for _ in range(size)]
    total = sum(samples)
    if total == 0:
        return [1.0 / size for _ in range(size)]
    return [value / total for value in samples]


def rollout(game, board, player, max_moves):
    current_board = list(board)
    current_player = player
    root_player = player
    moves_left = max_moves
    while True:
        outcome = game.check_winner(current_board, current_player)
        if outcome is not None:
            if outcome == 0:
                return 0.0
            return 1.0 if outcome == root_player else -1.0
        legal = game.legal_moves(current_board, current_player)
        if not legal:
            outcome = game.max_moves_result(current_board, current_player)
            if outcome == 0:
                return 0.0
            return 1.0 if outcome == root_player else -1.0
        move = random.choice(legal)
        current_board = game.apply_move(current_board, move, current_player)
        current_player = -current_player
        if moves_left:
            moves_left -= 1
            if moves_left == 0:
                outcome = game.max_moves_result(current_board, current_player)
                if outcome == 0:
                    return 0.0
                return 1.0 if outcome == root_player else -1.0


def select_child(node, c_puct):
    best_score = -float("inf")
    best_move = None
    best_child = None
    sqrt_visits = math.sqrt(node.visit_count)
    for move, child in node.children.items():
        score = -child.value() + c_puct * child.prior * sqrt_visits / (1 + child.visit_count)
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child
    return best_move, best_child


def mcts_search(
    game,
    board,
    player,
    simulations,
    c_puct,
    dirichlet_alpha,
    noise_fraction,
    max_rollout_moves,
):
    root = Node(0.0)
    valid_moves = game.legal_moves(board, player)
    if not valid_moves:
        return [0.0 for _ in range(game.action_size)], [0 for _ in range(game.action_size)]

    priors = [1.0 / len(valid_moves) for _ in valid_moves]
    if noise_fraction > 0:
        noise = _dirichlet(dirichlet_alpha, len(valid_moves))
        priors = [
            (1 - noise_fraction) * prior + noise_fraction * noise_value
            for prior, noise_value in zip(priors, noise)
        ]

    for move, prior in zip(valid_moves, priors):
        root.children[move] = Node(prior)
    root.visit_count = 1

    for _ in range(simulations):
        node = root
        search_path = [node]
        current_board = list(board)
        current_player = player
        outcome = None

        while node.children:
            move, node = select_child(node, c_puct)
            current_board = game.apply_move(current_board, move, current_player)
            current_player = -current_player
            search_path.append(node)
            outcome = game.check_winner(current_board, current_player)
            if outcome is not None:
                break

        if outcome is None:
            valid = game.legal_moves(current_board, current_player)
            if valid:
                prior = 1.0 / len(valid)
                for move in valid:
                    node.children[move] = Node(prior)
            value = rollout(game, current_board, current_player, max_rollout_moves)
        else:
            if outcome == 0:
                value = 0.0
            elif outcome == current_player:
                value = 1.0
            else:
                value = -1.0

        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value

    visits = [0 for _ in range(game.action_size)]
    for move, child in root.children.items():
        visits[move] = child.visit_count
    total = sum(visits)
    if total == 0:
        policy = [0.0 for _ in range(game.action_size)]
    else:
        policy = [count / total for count in visits]
    return policy, visits


def play_game(game, simulations, ai_player, c_puct, dirichlet_alpha, noise_fraction, max_moves):
    board = game.initial_board()
    player = 1

    while True:
        if player == ai_player:
            policy, _ = mcts_search(
                game,
                board,
                player,
                simulations,
                c_puct,
                dirichlet_alpha,
                noise_fraction,
                max_moves,
            )
            valid = game.legal_moves(board, player)
            move = max(valid, key=lambda idx: policy[idx])
        else:
            move = random.choice(game.legal_moves(board, player))
        board = game.apply_move(board, move, player)
        outcome = game.check_winner(board, -player)
        if outcome is not None:
            return outcome
        player = -player
        if max_moves:
            max_moves -= 1
            if max_moves == 0:
                return game.max_moves_result(board, player)


def evaluate(game, games, simulations, c_puct, dirichlet_alpha, noise_fraction, max_moves):
    wins = 0
    draws = 0
    losses = 0

    for idx in range(games):
        ai_player = 1 if idx % 2 == 0 else -1
        outcome = play_game(game, simulations, ai_player, c_puct, dirichlet_alpha, noise_fraction, max_moves)
        if outcome == 0:
            draws += 1
        elif outcome == ai_player:
            wins += 1
        else:
            losses += 1

    win_rate = (wins + 0.5 * draws) / games
    return win_rate, wins, draws, losses


def elo_from_win_rate(win_rate, opponent_elo=0.0):
    win_rate = min(max(win_rate, 0.001), 0.999)
    diff = 400.0 * math.log10(win_rate / (1.0 - win_rate))
    return opponent_elo + diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="tictactoe")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--noise-fraction", type=float, default=0.25)
    args = parser.parse_args()

    random.seed(args.seed)
    game = get_game(args.game)
    max_moves = game.max_moves
    win_rate, wins, draws, losses = evaluate(
        game,
        args.games,
        args.simulations,
        args.c_puct,
        args.dirichlet_alpha,
        args.noise_fraction,
        max_moves,
    )
    elo = elo_from_win_rate(win_rate)
    print(
        f"MCTS vs random: win_rate={win_rate:.3f} Elo={elo:.1f} "
        f"(wins={wins} draws={draws} losses={losses})"
    )


if __name__ == "__main__":
    main()
