const boardEl = document.querySelector("#board");
const statusEl = document.querySelector("#status");
const restartBtn = document.querySelector("#restart");
const refreshBtn = document.querySelector("#refresh-charts");
const gameSelect = document.querySelector("#game-select");
const noteEl = document.querySelector("#note");

let board = [];
let initialBoard = [];
let gameName = "tictactoe";
let gridSize = 3;
let gameOver = false;
let aiThinking = false;
let selectedIndex = null;
let moveCount = 0;
let maxMoves = 0;
let cells = [];

function setStatus(text) {
  statusEl.textContent = text;
}

function pieceChar(value) {
  if (value === 0) return "";
  if (gameName === "tictactoe") return value === 1 ? "X" : "O";
  if (gameName === "go5") return value === 1 ? "B" : "W";
  const map = { 1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K" };
  const base = map[Math.abs(value)] || "?";
  return value > 0 ? base : base.toLowerCase();
}

function buildBoard() {
  boardEl.innerHTML = "";
  boardEl.style.setProperty("--grid-size", gridSize);
  cells = [];
  board.forEach((_, idx) => {
    const cell = document.createElement("div");
    cell.className = "cell";
    cell.dataset.index = String(idx);
    cell.addEventListener("click", handleCellClick);
    boardEl.appendChild(cell);
    cells.push(cell);
  });
}

function renderBoard() {
  if (cells.length !== board.length) {
    buildBoard();
  }
  cells.forEach((cell, idx) => {
    const value = board[idx];
    cell.textContent = pieceChar(value);
    cell.classList.toggle("is-x", gameName === "tictactoe" && value === 1);
    cell.classList.toggle("is-o", gameName === "tictactoe" && value === -1);
    cell.classList.toggle("is-pos", gameName !== "tictactoe" && value > 0);
    cell.classList.toggle("is-neg", gameName !== "tictactoe" && value < 0);
    cell.classList.toggle("selected", idx === selectedIndex);
  });
}

function updateNote() {
  if (gameName === "tictactoe") {
    noteEl.textContent = "You are X. AI is O.";
  } else if (gameName === "go5") {
    noteEl.textContent = "You are B. AI is W.";
  } else {
    noteEl.textContent = "You are White. Select a piece, then a destination.";
  }
}

function endGame(winner) {
  gameOver = true;
  if (winner === 1) {
    setStatus("You win.");
  } else if (winner === -1) {
    setStatus("AI wins.");
  } else {
    setStatus("Draw.");
  }
}

async function sendMove(move) {
  aiThinking = true;
  setStatus("AI thinking...");
  try {
    const response = await fetch("/api/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ board, move, move_count: moveCount }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Move failed");
    }
    board = data.board;
    moveCount = data.move_count ?? moveCount;
    renderBoard();
    const winner = data.winner;
    if (winner !== null && winner !== undefined) {
      endGame(winner);
    } else {
      setStatus("Your move.");
    }
  } catch (err) {
    setStatus(err.message || "Move failed.");
  } finally {
    aiThinking = false;
  }
}

function handleCellClick(event) {
  const idx = Number(event.currentTarget.dataset.index);
  if (gameOver || aiThinking) {
    return;
  }

  if (gameName === "chess5") {
    if (selectedIndex === null) {
      if (board[idx] > 0) {
        selectedIndex = idx;
        renderBoard();
      }
      return;
    }
    if (idx === selectedIndex) {
      selectedIndex = null;
      renderBoard();
      return;
    }
    const move = selectedIndex * board.length + idx;
    selectedIndex = null;
    renderBoard();
    sendMove(move);
    return;
  }

  if (board[idx] !== 0) {
    return;
  }
  sendMove(idx);
}

async function loadGameConfig(nextGame = null) {
  const options = nextGame
    ? {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ game: nextGame }),
      }
    : { method: "GET" };
  const response = await fetch("/api/game", options);
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.error || "Failed to load game.");
    return;
  }

  gameName = data.game;
  initialBoard = data.initial_board || [];
  board = initialBoard.slice();
  gridSize = Math.sqrt(board.length);
  maxMoves = data.max_moves || 0;
  selectedIndex = null;
  moveCount = 0;
  gameOver = false;
  aiThinking = false;
  if (gameSelect.value !== gameName) {
    gameSelect.value = gameName;
  }
  updateNote();
  renderBoard();
  setStatus("Your move.");
}

function restartGame() {
  board = initialBoard.slice();
  selectedIndex = null;
  moveCount = 0;
  gameOver = false;
  aiThinking = false;
  renderBoard();
  setStatus("Your move.");
}

function refreshCharts() {
  const images = document.querySelectorAll(".chart img");
  images.forEach((img) => {
    const base = img.dataset.src;
    img.src = `${base}?t=${Date.now()}`;
  });
}

gameSelect.addEventListener("change", () => loadGameConfig(gameSelect.value));
restartBtn.addEventListener("click", restartGame);
refreshBtn.addEventListener("click", refreshCharts);

loadGameConfig();
