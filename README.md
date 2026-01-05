# 井字棋 EfficientZero Demo

一个使用简化 EfficientZero 算法的井字棋强化学习网页 Demo。后端使用 Flask + PyTorch 进行自博弈训练，前端可以直接在浏览器中对战并触发训练。

## 运行

```bash
pip install -r requirements.txt
python app.py
```

然后访问 `http://localhost:5000`。

## 功能
- 自博弈训练：选择局数与 MCTS 模拟步数即可触发训练。
- 状态查看：展示累计训练步数与最近一次训练损失。
- 对战体验：在浏览器中直接与 EfficientZero 代理对战。
