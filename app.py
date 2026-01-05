from __future__ import annotations
import json
from threading import Lock
from typing import Any, Dict, List

from flask import Flask, jsonify, request, send_from_directory
import numpy as np

from efficient_zero import agent
from tic_tac_toe import TicTacToe

app = Flask(__name__, static_folder="static", static_url_path="")

training_lock = Lock()
training_history: List[Dict[str, Any]] = []


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status")
def status():
    last = training_history[-1] if training_history else None
    return jsonify({
        "training_steps": agent.training_steps,
        "last_session": last,
    })


@app.route("/api/train", methods=["POST"])
def train():
    payload = request.get_json(force=True)
    num_games = int(payload.get("games", 8))
    simulations = int(payload.get("simulations", 40))
    batch_size = int(payload.get("batch_size", 32))
    with training_lock:
        stats = agent.train_self_play(num_games=num_games, simulations=simulations, batch_size=batch_size)
        training_history.append(stats)
    return jsonify(stats)


def _board_from_payload(board_payload: List[int]) -> np.ndarray:
    flattened = list(board_payload)[:9]
    flattened += [0] * (9 - len(flattened))
    return np.array([int(v) for v in flattened], dtype=np.int8)


@app.route("/api/move", methods=["POST"])
def move():
    payload = request.get_json(force=True)
    board = _board_from_payload(payload.get("board", [0] * 9))
    player = int(payload.get("player", 1))
    env = TicTacToe()
    env.board = board
    action, visit_probs = agent.select_action(env, player)
    return jsonify({
        "action": action,
        "policy": visit_probs,
    })


@app.route("/api/recommendations", methods=["POST"])
def recommendations():
    payload = request.get_json(force=True)
    board = _board_from_payload(payload.get("board", [0] * 9))
    player = int(payload.get("player", 1))
    env = TicTacToe()
    env.board = board
    _, visit_probs = agent.select_action(env, player)
    return jsonify({"policy": visit_probs})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
