"""Flask app for playing against the trained AI."""

import os
from threading import Lock

import torch
from flask import Flask, jsonify, render_template, request, send_from_directory

from games import get_game
from mcts import MCTS
from model import GameNet, load_model

SIMULATIONS = int(os.environ.get("MCTS_SIMULATIONS", "200"))

app = Flask(__name__)
STATE_LOCK = Lock()
DEVICE = torch.device("cpu")
STATE = {"game_name": None, "game": None, "model": None}


def load_or_init_model(game_name, game):
    model_path = os.path.join("artifacts", f"{game_name}_model.pth")
    if os.path.exists(model_path):
        model = load_model(
            model_path,
            device=DEVICE,
            board_size=game.board_size,
            action_size=game.action_size,
        )
    else:
        model = GameNet(board_size=game.board_size, action_size=game.action_size).to(DEVICE)
        model.eval()
    return model


def set_game(game_name):
    game = get_game(game_name)
    model = load_or_init_model(game_name, game)
    STATE.update({"game_name": game_name, "game": game, "model": model})
    return game


with STATE_LOCK:
    set_game(os.environ.get("GAME_NAME", "chess5"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/game", methods=["GET", "POST"])
def api_game():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        game_name = data.get("game")
        if not isinstance(game_name, str):
            return jsonify({"error": "Invalid game."}), 400
        try:
            with STATE_LOCK:
                set_game(game_name)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    with STATE_LOCK:
        game = STATE["game"]
        payload = {
            "game": STATE["game_name"],
            "board_size": game.board_size,
            "action_size": game.action_size,
            "initial_board": game.initial_board(),
            "max_moves": game.max_moves,
        }
    return jsonify(payload)


@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.get_json(silent=True) or {}
    board = data.get("board")
    move = data.get("move")
    move_count = data.get("move_count", 0)

    if not isinstance(board, list):
        return jsonify({"error": "Invalid board."}), 400
    if not isinstance(move, int):
        return jsonify({"error": "Invalid move."}), 400
    if not isinstance(move_count, int) or move_count < 0:
        return jsonify({"error": "Invalid move_count."}), 400

    with STATE_LOCK:
        game = STATE["game"]
        model = STATE["model"]

    if len(board) != game.board_size:
        return jsonify({"error": "Invalid board size."}), 400

    human_player = 1
    ai_player = -1

    legal = game.legal_moves(board, human_player)
    if move not in legal:
        return jsonify({"error": "Illegal move."}), 400

    board = game.apply_move(board, move, human_player)
    move_count += 1
    winner = game.check_winner(board, ai_player)
    if winner is not None:
        return jsonify({"board": board, "winner": winner, "move": None, "move_count": move_count})
    if game.max_moves and move_count >= game.max_moves:
        winner = game.max_moves_result(board, ai_player)
        return jsonify({"board": board, "winner": winner, "move": None, "move_count": move_count})

    legal = game.legal_moves(board, ai_player)
    if not legal:
        winner = game.check_winner(board, ai_player)
        if winner is None:
            winner = game.max_moves_result(board, ai_player)
        return jsonify({"board": board, "winner": winner, "move": None, "move_count": move_count})

    mcts = MCTS(game, model, simulations=SIMULATIONS, device=DEVICE)
    policy, _ = mcts.run(board, ai_player)
    ai_move = max(legal, key=lambda idx: policy[idx])
    board = game.apply_move(board, ai_move, ai_player)
    move_count += 1
    winner = game.check_winner(board, human_player)
    if winner is None and game.max_moves and move_count >= game.max_moves:
        winner = game.max_moves_result(board, human_player)
    return jsonify({"board": board, "move": ai_move, "winner": winner, "move_count": move_count})


@app.route("/artifacts/<path:filename>")
def artifacts(filename):
    return send_from_directory("artifacts", filename)


if __name__ == "__main__":
    app.run(debug=True)
