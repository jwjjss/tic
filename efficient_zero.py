from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tic_tac_toe import TicTacToe


class MiniNetwork(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.representation = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.ReLU(),
        )
        self.dynamics = nn.GRUCell(9, hidden_size)
        self.policy_head = nn.Linear(hidden_size, 9)
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_inference(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.representation(obs)
        policy = self.policy_head(hidden)
        value = torch.tanh(self.value_head(hidden))
        reward = torch.zeros_like(value)
        return value, reward, policy, hidden

    def recurrent_inference(self, hidden: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        next_hidden = self.dynamics(action, hidden)
        policy = self.policy_head(next_hidden)
        value = torch.tanh(self.value_head(next_hidden))
        reward = torch.zeros_like(value)
        return value, reward, policy, next_hidden


@dataclass
class Node:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "Node"] = None
    hidden_state: torch.Tensor = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class EfficientZeroAgent:
    """Tiny EfficientZero-inspired agent for Tic Tac Toe.

    The implementation is intentionally lightweight: it keeps a small GRU-based
    network and uses MCTS for action selection. Training is performed through
    repeated self-play games followed by policy/value learning.
    """

    MINIMAX_ELO = 1200.0

    def __init__(self, device: str | torch.device = "cpu", hidden_size: int = 64):
        self.device = torch.device(device)
        self.network = MiniNetwork(hidden_size=hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4, weight_decay=1e-5)
        self.training_steps = 0
        self.elo = 1000.0

    def select_action(self, env: TicTacToe, player: int, simulations: int = 50) -> Tuple[int, List[float]]:
        obs = torch.from_numpy(env.observation(player)).float().to(self.device)
        value, reward, policy_logits, hidden = self.network.initial_inference(obs)
        root = Node(prior=1.0, hidden_state=hidden.detach())
        policy = torch.softmax(policy_logits, dim=0).detach().cpu().numpy()
        for action, p in enumerate(policy):
            if env.board[action] == 0:
                root.children[action] = Node(prior=float(p))

        for _ in range(simulations):
            self._run_mcts(env.clone(), player, root)

        visits = np.array([root.children[a].visit_count if a in root.children else 0 for a in range(9)], dtype=np.float32)
        if visits.sum() == 0:
            visits = np.ones_like(visits)
        action = int(np.random.choice(range(9), p=visits / visits.sum()))
        return action, (visits / visits.sum()).tolist()

    def _run_mcts(self, env: TicTacToe, player: int, root: Node, discount: float = 0.997):
        node = root
        current_player = player
        search_path = [node]
        actions_taken: List[int] = []

        while node.children:
            max_score = -float("inf")
            best_action = None
            total_visits = sum(child.visit_count for child in node.children.values()) + 1
            for action, child in node.children.items():
                ucb = self._ucb_score(total_visits, child)
                if ucb > max_score:
                    max_score = ucb
                    best_action = action
            action = best_action
            env.apply_action(action, current_player)
            current_player = TicTacToe.opponent(current_player)
            node = node.children[action]
            search_path.append(node)
            actions_taken.append(action)
            winner = env.check_winner()
            if winner is not None:
                break

        # Expand
        winner = env.check_winner()
        if winner is None:
            obs = torch.from_numpy(env.observation(current_player)).float().to(self.device)
            value, reward, policy_logits, hidden = self.network.initial_inference(obs)
            policy = torch.softmax(policy_logits, dim=0).detach().cpu().numpy()
            node.hidden_state = hidden.detach()
            for action, p in enumerate(policy):
                if env.board[action] == 0:
                    node.children[action] = Node(prior=float(p))
            leaf_value = float(value.item())
        else:
            leaf_value = float(winner)

        # Backpropagate
        for step, node in enumerate(reversed(search_path)):
            node.visit_count += 1
            node.value_sum += (discount ** step) * leaf_value
            leaf_value = -leaf_value  # switch perspective

    @staticmethod
    def _ucb_score(total_visit_count: int, child: Node, pb_c_base: float = 1.25, pb_c_init: float = 1.5) -> float:
        pb_c = math.log((total_visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
        pb_c *= math.sqrt(total_visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior
        value_score = child.value()
        return prior_score + value_score

    def self_play(self, num_games: int = 20, simulations: int = 50) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
        dataset: List[Dict] = []
        replays: List[Dict] = []
        outcomes = {"wins": 0, "losses": 0, "draws": 0}

        for _ in range(num_games):
            env = TicTacToe()
            player = 1
            game_history: List[Dict] = []
            moves: List[Dict] = []
            while True:
                observation = env.observation(player)
                action, visit_probs = self.select_action(env, player, simulations=simulations)
                moves.append({
                    "player": player,
                    "board": env.board.astype(int).tolist(),
                    "action": action,
                    "policy": visit_probs,
                })
                game_history.append({
                    "observation": observation,
                    "player": player,
                    "action_probs": visit_probs,
                })
                env.apply_action(action, player)
                winner = env.check_winner()
                if winner is not None:
                    for step, item in enumerate(game_history):
                        value = winner * item["player"]
                        dataset.append({
                            "observation": item["observation"],
                            "policy_target": np.array(item["action_probs"], dtype=np.float32),
                            "value_target": float(value),
                        })
                    if winner == 1:
                        outcomes["wins"] += 1
                    elif winner == -1:
                        outcomes["losses"] += 1
                    else:
                        outcomes["draws"] += 1
                    replays.append({"moves": moves, "winner": int(winner)})
                    break
                player = TicTacToe.opponent(player)
        return dataset, replays, outcomes

    def _minimax(self, env: TicTacToe, player: int, alpha: float = -float("inf"), beta: float = float("inf")) -> float:
        winner = env.check_winner()
        if winner is not None:
            return float(winner * player)
        best_score = -float("inf")
        for action in env.legal_actions():
            next_env = env.clone()
            next_env.apply_action(action, player)
            score = -self._minimax(next_env, TicTacToe.opponent(player), -beta, -alpha)
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score

    def _minimax_action(self, env: TicTacToe, player: int) -> int:
        best_score = -float("inf")
        best_actions: List[int] = []
        for action in env.legal_actions():
            next_env = env.clone()
            next_env.apply_action(action, player)
            score = -self._minimax(next_env, TicTacToe.opponent(player))
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
        return random.choice(best_actions)

    def evaluate_vs_minimax(self, num_games: int = 10, simulations: int = 50) -> Dict[str, int]:
        outcomes = {"wins": 0, "losses": 0, "draws": 0}
        for game_index in range(num_games):
            env = TicTacToe()
            agent_player = 1 if game_index % 2 == 0 else -1
            current_player = 1
            while True:
                if current_player == agent_player:
                    action, _ = self.select_action(env, current_player, simulations=simulations)
                else:
                    action = self._minimax_action(env, current_player)
                env.apply_action(action, current_player)
                winner = env.check_winner()
                if winner is not None:
                    if winner == agent_player:
                        outcomes["wins"] += 1
                    elif winner == 0:
                        outcomes["draws"] += 1
                    else:
                        outcomes["losses"] += 1
                    break
                current_player = TicTacToe.opponent(current_player)
        return outcomes

    def train_step(self, batch: List[Dict]):
        obs_batch = torch.tensor(np.stack([b["observation"] for b in batch]), dtype=torch.float32, device=self.device)
        policy_targets = torch.tensor(np.stack([b["policy_target"] for b in batch]), dtype=torch.float32, device=self.device)
        value_targets = torch.tensor([b["value_target"] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)

        value, reward, policy_logits, _ = self.network.initial_inference(obs_batch)
        policy_loss = -(policy_targets * torch.log_softmax(policy_logits, dim=-1)).sum(dim=1).mean()
        value_loss = nn.functional.smooth_l1_loss(value, value_targets)
        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.training_steps += 1
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }

    def train_self_play(self, num_games: int = 10, simulations: int = 40, batch_size: int = 32) -> Dict:
        dataset, replays, outcomes = self.self_play(num_games=num_games, simulations=simulations)
        random.shuffle(dataset)
        metrics = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            metrics.append(self.train_step(batch))
        avg_loss = sum(m["loss"] for m in metrics) / max(1, len(metrics))
        elo_before = self.elo
        eval_outcomes = self.evaluate_vs_minimax(num_games=max(4, num_games // 2), simulations=simulations)
        self._update_elo(eval_outcomes)
        return {
            "games": num_games,
            "samples": len(dataset),
            "avg_loss": avg_loss,
            "steps": self.training_steps,
            "wins": outcomes["wins"],
            "losses": outcomes["losses"],
            "draws": outcomes["draws"],
            "eval_wins": eval_outcomes["wins"],
            "eval_losses": eval_outcomes["losses"],
            "eval_draws": eval_outcomes["draws"],
            "elo": self.elo,
            "elo_change": self.elo - elo_before,
            "replays": replays,
        }

    def _update_elo(self, outcomes: Dict[str, int], k: float = 24.0):
        total = outcomes["wins"] + outcomes["losses"] + outcomes["draws"]
        if total == 0:
            return
        score = (outcomes["wins"] + 0.5 * outcomes["draws"]) / total
        expected = 1.0 / (1.0 + 10 ** ((self.MINIMAX_ELO - self.elo) / 400.0))
        self.elo += k * (score - expected)

    def to_cpu(self):
        self.network.to("cpu")
        self.device = torch.device("cpu")


# Shared singleton agent for the web app
agent = EfficientZeroAgent()
