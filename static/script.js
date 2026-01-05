const boardEl = document.getElementById('board');
const messageEl = document.getElementById('message');
const statusEl = document.getElementById('status');
const eloEl = document.getElementById('elo-display');
const trainButton = document.getElementById('train');
const resetButton = document.getElementById('reset');
const agentMoveButton = document.getElementById('agent-move');
const gamesInput = document.getElementById('games');
const simsInput = document.getElementById('sims');
const replaySelect = document.getElementById('replay-select');
const replayPlayButton = document.getElementById('replay-play');
const replayStopButton = document.getElementById('replay-stop');
const replayBoardEl = document.getElementById('replay-board');
const replayMetaEl = document.getElementById('replay-meta');

let board = Array(9).fill(0);
let currentPlayer = 1; // 1 = X (玩家), -1 = O (AI)
let replayBoard = Array(9).fill(0);
let replays = [];
let replayTimer = null;
let currentReplayIndex = 0;

function renderBoard() {
  boardEl.innerHTML = '';
  board.forEach((cell, idx) => {
    const div = document.createElement('div');
    div.className = 'cell';
    div.textContent = cell === 1 ? 'X' : cell === -1 ? 'O' : '';
    div.onclick = () => handlePlayerMove(idx);
    boardEl.appendChild(div);
  });
}

function renderReplayBoard() {
  replayBoardEl.innerHTML = '';
  replayBoard.forEach((cell) => {
    const div = document.createElement('div');
    div.className = 'cell';
    div.textContent = cell === 1 ? 'X' : cell === -1 ? 'O' : '';
    replayBoardEl.appendChild(div);
  });
}

function checkWinner() {
  const lines = [
    [0,1,2],[3,4,5],[6,7,8],
    [0,3,6],[1,4,7],[2,5,8],
    [0,4,8],[2,4,6]
  ];
  for (const [a,b,c] of lines) {
    const s = board[a] + board[b] + board[c];
    if (s === 3) return 1;
    if (s === -3) return -1;
  }
  if (board.every(v => v !== 0)) return 0;
  return null;
}

function setMessage(text) {
  messageEl.textContent = text;
}

function handlePlayerMove(idx) {
  if (board[idx] !== 0) return;
  if (checkWinner() !== null) return;
  board[idx] = currentPlayer;
  renderBoard();
  const winner = checkWinner();
  if (winner !== null) return endGame(winner);
}

async function agentMove() {
  if (checkWinner() !== null) return;
  agentMoveButton.disabled = true;
  setMessage('AI 正在搜索...');
  const res = await fetch('/api/move', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ board, player: -1 }),
  });
  const data = await res.json();
  board[data.action] = -1;
  renderBoard();
  agentMoveButton.disabled = false;
  const winner = checkWinner();
  if (winner !== null) endGame(winner);
  else setMessage('轮到你了');
}

function endGame(winner) {
  if (winner === 1) setMessage('你赢了！');
  else if (winner === -1) setMessage('AI 获胜');
  else setMessage('平局');
}

function resetBoard() {
  board = Array(9).fill(0);
  currentPlayer = 1;
  setMessage('开始对局，X 先手');
  renderBoard();
}

async function loadStatus() {
  const res = await fetch('/api/status');
  const data = await res.json();
  statusEl.textContent = `训练步数：${data.training_steps || 0}${data.last_session ? `，最近一次平均损失 ${data.last_session.avg_loss.toFixed(4)}` : ''}`;
  if (data.elo !== undefined) {
    eloEl.textContent = `ELO: ${data.elo.toFixed(1)}`;
  }
}

async function train() {
  trainButton.disabled = true;
  trainButton.textContent = '训练中...';
  document.getElementById('train-status').innerHTML = '<div class="progress"></div>';
  const res = await fetch('/api/train', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      games: Number(gamesInput.value),
      simulations: Number(simsInput.value),
    }),
  });
  const data = await res.json();
  trainButton.disabled = false;
  trainButton.textContent = '开始训练';
  document.getElementById('train-status').textContent = `完成 ${data.games} 局自博弈，平均损失 ${data.avg_loss.toFixed(4)}`;
  await loadStatus();
  await loadReplays();
}

function renderReplayOptions() {
  replaySelect.innerHTML = '';
  if (!replays.length) {
    const opt = document.createElement('option');
    opt.textContent = '暂无回放';
    opt.value = '';
    replaySelect.appendChild(opt);
    replayPlayButton.disabled = true;
    replayStopButton.disabled = true;
    replayMetaEl.textContent = '';
    replayBoard = Array(9).fill(0);
    renderReplayBoard();
    return;
  }
  replayPlayButton.disabled = false;
  replayStopButton.disabled = false;
  replays
    .map((r, idx) => ({ replay: r, idx }))
    .sort((a, b) => (a.replay.id || 0) - (b.replay.id || 0))
    .forEach(({ replay, idx }) => {
      const opt = document.createElement('option');
      opt.value = idx;
      const outcome = replay.winner === 1 ? 'X 胜' : replay.winner === -1 ? 'O 胜' : '平局';
      const timeLabel = replay.timestamp ? new Date(replay.timestamp * 1000).toLocaleTimeString() : `对局 ${idx + 1}`;
      opt.textContent = `${timeLabel} · ${outcome}`;
      replaySelect.appendChild(opt);
    });
  currentReplayIndex = Number(replaySelect.value) || 0;
  replayMetaEl.textContent = `共 ${replays.length} 盘`;
}

function stopReplay() {
  if (replayTimer) {
    clearInterval(replayTimer);
    replayTimer = null;
  }
}

function playReplay() {
  stopReplay();
  if (!replays.length) return;
  currentReplayIndex = Number(replaySelect.value) || 0;
  const replay = replays[currentReplayIndex];
  if (!replay || !replay.moves) return;
  let step = 0;
  replayBoard = Array(9).fill(0);
  renderReplayBoard();
  replayMetaEl.textContent = `播放中：共 ${replay.moves.length} 手`;
  replayTimer = setInterval(() => {
    if (step >= replay.moves.length) {
      stopReplay();
      const result = replay.winner === 1 ? 'X 获胜' : replay.winner === -1 ? 'O 获胜' : '平局';
      replayMetaEl.textContent = `完成：${result}`;
      return;
    }
    const move = replay.moves[step];
    replayBoard = Array.from(move.board);
    replayBoard[move.action] = move.player;
    renderReplayBoard();
    step += 1;
  }, 700);
}

async function loadReplays() {
  const res = await fetch('/api/replays');
  const data = await res.json();
  replays = data.replays || [];
  if (data.elo !== undefined) {
    eloEl.textContent = `ELO: ${data.elo.toFixed(1)}`;
  }
  renderReplayOptions();
}

trainButton.onclick = train;
resetButton.onclick = resetBoard;
agentMoveButton.onclick = agentMove;
replayPlayButton.onclick = playReplay;
replayStopButton.onclick = () => {
  stopReplay();
  replayMetaEl.textContent = '已停止';
};
replaySelect.onchange = () => {
  stopReplay();
  currentReplayIndex = Number(replaySelect.value) || 0;
};

resetBoard();
renderReplayBoard();
loadStatus();
loadReplays();
setInterval(loadStatus, 6000);
