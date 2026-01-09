# Tic-Tac-Toe RL + MCTS

This project trains a Tic-Tac-Toe agent with self-play, MCTS, and a policy/value network, then serves a web UI to play against the trained model. Training logs include win rate and Elo vs a random bot, plus plots.

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Train

```bash
python training.py --game tictactoe --iterations 30 --games-per-iteration 50 --simulations 100 --eval-interval 5 --eval-games 1000
```

Supported games:
- `tictactoe` (3x3)
- `go5` (5x5 Go, no pass, 50-move limit uses liberties to decide)
- `chess5` (5x5 chess, no castling/en passant, check rules enforced, 50-move draw)

To test with minimax self-play (tictactoe only, evaluation always uses the neural net vs random):

```bash
python training.py --game tictactoe --search minimax
```

Artifacts saved to `artifacts/`:
- `{game}_model.pth` for all games (trained weights)
- `training_log.csv`
- `win_rate_curve.png`
- `elo_curve.png`

## Run Web App

```bash
python app.py
```

Open `http://127.0.0.1:5000` in a browser. Use the dropdown to switch games; the UI will hot-load `artifacts/{game}_model.pth`.

## Tests

```bash
pytest -q
```

## Notes

- The Elo is computed from win rate vs the random bot (random Elo = 0).
- Increase `--simulations` for stronger play at the cost of speed.
- Set `MCTS_SIMULATIONS=200` to change the web AI strength.
