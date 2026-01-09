"""Monte Carlo Tree Search guided by a policy/value network."""

import math
import random

import torch


class Node:
    def __init__(self, prior):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    def expanded(self):
        return bool(self.children)

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, game, model, simulations=100, c_puct=1.5, device=None):
        self.game = game
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = device or torch.device("cpu")

    def run(
        self,
        board,
        player,
        add_noise=False,
        dirichlet_alpha=0.3,
        noise_fraction=0.25,
    ):
        root = Node(0.0)
        policy, _ = self._evaluate(board, player)
        valid_moves = self.game.legal_moves(board, player)
        if add_noise and valid_moves:
            noise = _dirichlet(dirichlet_alpha, len(valid_moves))
            for move, noise_value in zip(valid_moves, noise):
                policy[move] = (1 - noise_fraction) * policy[move] + noise_fraction * noise_value
        for move in valid_moves:
            root.children[move] = Node(policy[move])
        root.visit_count = 1

        for _ in range(self.simulations):
            node = root
            search_path = [node]
            current_board = list(board)
            current_player = player
            outcome = None

            while node.expanded():
                move, node = self._select_child(node)
                current_board = self.game.apply_move(current_board, move, current_player)
                current_player = -current_player
                search_path.append(node)
                outcome = self.game.check_winner(current_board, current_player)
                if outcome is not None:
                    break

            if outcome is None:
                policy, value = self._evaluate(current_board, current_player)
                for move in self.game.legal_moves(current_board, current_player):
                    node.children[move] = Node(policy[move])
            else:
                if outcome == 0:
                    value = 0.0
                elif outcome == current_player:
                    value = 1.0
                else:
                    value = -1.0

            # value is always from the current player's perspective, so flip when moving up.
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                value = -value

        visits = [0 for _ in range(self.game.action_size)]
        for move, child in root.children.items():
            visits[move] = child.visit_count
        total = sum(visits)
        if total == 0:
            valid = self.game.legal_moves(board, player)
            action_size = self.game.action_size
            if not valid:
                policy = [0.0 for _ in range(action_size)]
            else:
                policy = [1.0 / len(valid) if idx in valid else 0.0 for idx in range(action_size)]
        else:
            policy = [count / total for count in visits]
        return policy, visits

    def _evaluate(self, board, player):
        encoded = self.game.encode(board, player)
        inp = torch.tensor([encoded], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            policy_logits, value = self.model(inp)
        policy_logits = policy_logits[0]
        legal_moves = self.game.legal_moves(board, player)
        mask = torch.full((self.game.action_size,), -1e9, device=self.device)
        if legal_moves:
            mask[torch.tensor(legal_moves, device=self.device)] = 0.0
        masked_logits = policy_logits + mask
        policy = torch.softmax(masked_logits, dim=-1).cpu().tolist()
        return policy, float(value.item())

    def _select_child(self, node):
        best_score = -float("inf")
        best_move = None
        best_child = None
        sqrt_visits = math.sqrt(node.visit_count)
        for move, child in node.children.items():
            # child.value() is from the child's player perspective; negate to align with this node.
            score = -child.value() + self.c_puct * child.prior * sqrt_visits / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child


def _dirichlet(alpha, size):
    samples = [random.gammavariate(alpha, 1.0) for _ in range(size)]
    total = sum(samples)
    if total == 0:
        return [1.0 / size for _ in range(size)]
    return [value / total for value in samples]
