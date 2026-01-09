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

function isChessGame() {
  return gameName === "chess5";
}

function displayIndexToBoardIndex(displayIndex) {
  if (!isChessGame()) {
    return displayIndex;
  }
  const row = Math.floor(displayIndex / gridSize);
  const col = displayIndex % gridSize;
  const boardRow = gridSize - 1 - row;
  return boardRow * gridSize + col;
}

function setStatus(text) {
  statusEl.textContent = text;
}

function pieceChar(value) {
  if (value === 0) return "";
  if (gameName === "tictactoe") return value === 1 ? "X" : "O";
  if (gameName === "go5") return value === 1 ? "B" : "W";
  if (gameName === "chess5") {
    const white = { 1: "♙", 2: "♘", 3: "♗", 4: "♖", 5: "♕", 6: "♔" };
    const black = { 1: "♟", 2: "♞", 3: "♝", 4: "♜", 5: "♛", 6: "♚" };
    const base = Math.abs(value);
    return value > 0 ? white[base] || "?" : black[base] || "?";
  }
  return "?";
}

function buildBoard() {
  boardEl.innerHTML = "";
  boardEl.style.setProperty("--grid-size", gridSize);
  boardEl.dataset.game = gameName;
  boardEl.classList.toggle("board--chess", isChessGame());
  cells = [];
  for (let displayIndex = 0; displayIndex < board.length; displayIndex += 1) {
    const cell = document.createElement("div");
    cell.className = "cell";
    const row = Math.floor(displayIndex / gridSize);
    const col = displayIndex % gridSize;
    const boardIndex = displayIndexToBoardIndex(displayIndex);
    cell.dataset.index = String(boardIndex);
    if (isChessGame()) {
      cell.classList.add("cell--chess");
      cell.classList.toggle("cell--dark", (row + col) % 2 === 0);
      cell.classList.toggle("cell--light", (row + col) % 2 === 1);
    }
    cell.addEventListener("click", handleCellClick);
    boardEl.appendChild(cell);
    cells.push(cell);
  }
}

function renderBoard() {
  if (cells.length !== board.length || boardEl.dataset.game !== gameName) {
    buildBoard();
  }
  cells.forEach((cell) => {
    const boardIndex = Number(cell.dataset.index);
    const value = board[boardIndex];
    cell.textContent = pieceChar(value);
    cell.classList.toggle("is-x", gameName === "tictactoe" && value === 1);
    cell.classList.toggle("is-o", gameName === "tictactoe" && value === -1);
    cell.classList.toggle("is-pos", gameName !== "tictactoe" && value > 0);
    cell.classList.toggle("is-neg", gameName !== "tictactoe" && value < 0);
    cell.classList.toggle("selected", boardIndex === selectedIndex);
    cell.classList.toggle("has-piece", value !== 0);
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
  document.body.classList.toggle("game-chess", isChessGame());
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
