"""Self-play training loop with MCTS and Elo evaluation."""

import argparse
import csv
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import random

import matplotlib.pyplot as plt
import torch
from torch import optim

from games import get_game
from mcts import MCTS
from minimax import minimax_policy
from model import GameNet, save_model


def select_move_from_visits(visits, temperature):
    valid = [(idx, count) for idx, count in enumerate(visits) if count > 0]
    if not valid:
        return 0
    if temperature <= 1e-3:
        return max(valid, key=lambda item: item[1])[0]
    power = 1.0 / temperature
    adjusted = [count**power for _, count in valid]
    total = sum(adjusted)
    r = random.random() * total
    cumulative = 0.0
    for (idx, _), weight in zip(valid, adjusted):
        cumulative += weight
        if r <= cumulative:
            return idx
    return valid[-1][0]


def select_move_from_policy(policy, temperature):
    valid = [(idx, prob) for idx, prob in enumerate(policy) if prob > 0]
    if not valid:
        return 0
    if temperature <= 1e-3:
        return max(valid, key=lambda item: item[1])[0]
    power = 1.0 / temperature
    adjusted = [prob**power for _, prob in valid]
    total = sum(adjusted)
    r = random.random() * total
    cumulative = 0.0
    for (idx, _), weight in zip(valid, adjusted):
        cumulative += weight
        if r <= cumulative:
            return idx
    return valid[-1][0]


def model_policy(model, game, board, player, device):
    encoded = game.encode(board, player)
    inp = torch.tensor([encoded], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, _ = model(inp)
    logits = logits[0]
    legal_moves = game.legal_moves(board, player)
    mask = torch.full((game.action_size,), -1e9, device=device)
    if legal_moves:
        mask[torch.tensor(legal_moves, device=device)] = 0.0
    masked_logits = logits + mask
    policy = torch.softmax(masked_logits, dim=-1).cpu().tolist()
    return policy


def self_play_game(
    model,
    game,
    simulations,
    device,
    dirichlet_alpha=0.3,
    noise_fraction=0.25,
    temperature=1.0,
    temperature_moves=5,
    search_mode="mcts",
    max_moves=0,
):
    board = game.initial_board()
    player = 1
    history = []
    move_index = 0
    mcts = None
    if search_mode == "mcts":
        mcts = MCTS(game, model, simulations=simulations, device=device)

    while True:
        if search_mode == "minimax":
            if game.name != "tictactoe":
                raise ValueError("minimax is only supported for tictactoe")
            policy, _ = minimax_policy(board, player)
            temp = temperature if move_index < temperature_moves else 1e-3
            move = select_move_from_policy(policy, temp)
        else:
            policy, visits = mcts.run(
                board,
                player,
                add_noise=True,
                dirichlet_alpha=dirichlet_alpha,
                noise_fraction=noise_fraction,
            )
            temp = temperature if move_index < temperature_moves else 1e-3
            move = select_move_from_visits(visits, temp)
        history.append((game.encode(board, player), policy, player))
        board = game.apply_move(board, move, player)
        outcome = game.check_winner(board, -player)
        if outcome is not None:
            break
        player = -player
        move_index += 1
        if max_moves and move_index >= max_moves:
            outcome = game.max_moves_result(board, player)
            break

    data = []
    for state, policy, state_player in history:
        if outcome == 0:
            value = 0.0
        elif outcome == state_player:
            value = 1.0
        else:
            value = -1.0
        data.append((state, policy, value))
    return data


def train_on_data(model, data, device, batch_size=64, epochs=1, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    states = torch.tensor([item[0] for item in data], dtype=torch.float32, device=device)
    target_policies = torch.tensor([item[1] for item in data], dtype=torch.float32, device=device)
    target_values = torch.tensor([item[2] for item in data], dtype=torch.float32, device=device)

    dataset_size = states.size(0)
    indices = list(range(dataset_size))
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_steps = 0

    for _ in range(epochs):
        random.shuffle(indices)
        for start in range(0, dataset_size, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_states = states[batch_idx]
            batch_policies = target_policies[batch_idx]
            batch_values = target_values[batch_idx]

            optimizer.zero_grad()
            logits, values = model(batch_states)
            log_probs = torch.log_softmax(logits, dim=1)
            policy_loss = -(batch_policies * log_probs).sum(dim=1).mean()
            value_loss = torch.mean((values - batch_values) ** 2)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_steps += 1
    if total_steps == 0:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
    return {
        "loss": total_loss / total_steps,
        "policy_loss": total_policy_loss / total_steps,
        "value_loss": total_value_loss / total_steps,
    }


def play_vs_random(model, game, ai_player, device, max_moves=0):
    board = game.initial_board()
    player = 1

    while True:
        if player == ai_player:
            policy = model_policy(model, game, board, player, device)
            valid = game.legal_moves(board, player)
            move = max(valid, key=lambda idx: policy[idx])
        else:
            move = random.choice(game.legal_moves(board, player))
        board = game.apply_move(board, move, player)
        outcome = game.check_winner(board, -player)
        if outcome is not None:
            return outcome
        player = -player
        if max_moves and max_moves > 0:
            max_moves -= 1
            if max_moves == 0:
                return game.max_moves_result(board, player)


def elo_from_win_rate(win_rate, opponent_elo=0.0):
    win_rate = min(max(win_rate, 0.001), 0.999)
    diff = 400.0 * math.log10(win_rate / (1.0 - win_rate))
    return opponent_elo + diff


def evaluate(model, game, games, device, max_moves=0):
    wins = 0
    draws = 0
    losses = 0

    for idx in range(games):
        ai_player = 1 if idx % 2 == 0 else -1
        outcome = play_vs_random(model, game, ai_player, device, max_moves=max_moves)
        if outcome == 0:
            draws += 1
        elif outcome == ai_player:
            wins += 1
        else:
            losses += 1

    win_rate = (wins + 0.5 * draws) / games
    return win_rate, wins, draws, losses


def plot_metrics(log_rows, output_dir):
    iterations = [row[0] for row in log_rows]
    win_rates = [row[1] for row in log_rows]
    elos = [row[2] for row in log_rows]

    plt.figure(figsize=(8, 4))
    plt.plot(iterations, win_rates, marker="o")
    plt.ylim(0, 1)
    plt.xlabel("Iteration")
    plt.ylabel("Win rate vs random")
    plt.title("Win rate")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "win_rate_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(iterations, elos, marker="o", color="orange")
    plt.xlabel("Iteration")
    plt.ylabel("Elo")
    plt.title("Elo vs random")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "elo_curve.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="chess")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--games-per-iteration", type=int, default=256)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--search", choices=["mcts", "minimax"], default="mcts")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--noise-fraction", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-moves", type=int, default=5)
    args = parser.parse_args()
    selfplay_search = args.search
    eval_games = max(args.eval_games, 1000)
    game = get_game(args.game)
    max_moves = game.max_moves

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join("artifacts", f"{game.name}_model.pth")

    model = GameNet(
        board_size=game.board_size,
        action_size=game.action_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    log_path = os.path.join(args.output_dir, "training_log.csv")
    log_rows = []

    with open(log_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["iteration", "win_rate", "elo", "wins", "draws", "losses"])

        for iteration in range(1, args.iterations + 1):
            data = []
            model.eval()
            for game_idx in range(1, args.games_per_iteration + 1):
                data.extend(
                    self_play_game(
                        model,
                        game,
                        args.simulations,
                        device,
                        dirichlet_alpha=args.dirichlet_alpha,
                        noise_fraction=args.noise_fraction,
                        temperature=args.temperature,
                        temperature_moves=args.temperature_moves,
                        search_mode=selfplay_search,
                        max_moves=max_moves,
                    )
                )
                print(
                    f"Iteration {iteration}: self-play {game_idx}/{args.games_per_iteration}",
                    end="\r",
                    flush=True,
                )
            print()

            metrics = train_on_data(
                model,
                data,
                device,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
            )
            print(
                f"Iteration {iteration}: loss={metrics['loss']:.4f} "
                f"policy={metrics['policy_loss']:.4f} value={metrics['value_loss']:.4f}"
            )
            save_model(model, model_path)

            if iteration % args.eval_interval == 0:
                model.eval()
                win_rate, wins, draws, losses = evaluate(
                    model,
                    game,
                    eval_games,
                    device,
                    max_moves=max_moves,
                )
                elo = elo_from_win_rate(win_rate)
                writer.writerow([iteration, win_rate, elo, wins, draws, losses])
                csv_file.flush()
                log_rows.append((iteration, win_rate, elo))
                plot_metrics(log_rows, args.output_dir)
                print(
                    f"Iteration {iteration}: win rate={win_rate:.3f}, "
                    f"Elo={elo:.1f} (wins={wins} draws={draws} losses={losses})"
                )


if __name__ == "__main__":
    main()
