# 🇲🇲 Myanmar Checkers AI Visualizer (မြန်မာကျားကွက်)

An interactive full-stack web application featuring an authentic Traditional Myanmar Checkers game engine and Adversarial Search AI agents (Minimax with Alpha-Beta Pruning, Heuristic Evaluation, and Random Baseline).

## 🌟 Key Features
- **Authentic Traditional Myanmar Checkers Rules:**
  - 8×8 board, 32 playable dark squares, 12 pieces per player.
  - Regular Men: Forward-only diagonal movements and forward-only captures.
  - Flying King (ကျားကြီး/ဒမ်): Diagonal multi-square slides and captures across arbitrary distance.
  - Compulsory capture (မစားမနေရ) and Longest Line / Maximum Pieces rule.
- **Three AI Difficulty Levels:**
  - 🟢 **Easy:** `RandomAgent` (Uniform random baseline).
  - 🟡 **Medium:** `HeuristicAgent` (Greedy 1-step evaluation).
  - 🔴 **Hard:** `MinimaxAgent` (Adversarial Search with Depth 4 Lookahead, Alpha-Beta cutoffs, and move ordering).
- **Interactive Dual-Tab Adversarial Search Analytics:**
  - `[ 👤 Your Moves (White) ]`: Live evaluation of human legal options.
  - `[ 🤖 AI Decision (Black) ]`: Candidate branch evaluations, explored node metrics, and `AI CHOSE ⭐` indicator.
- **Automated Bot vs Bot Benchmark Simulation:**
  - Run 10 automated matches between different agent strategies to evaluate win rates and performance.

## 🚀 Local Run
```bash
pip install -r requirements.txt
python app.py
```
Open [http://localhost:5001](http://localhost:5001) in your browser.

## 🌐 Deploying to Render
1. Create a new **Web Service** on [Render](https://render.com).
2. Connect your `mm-checkers-ai-visualizer` repository.
3. Configure the settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
4. Click **Deploy Web Service**!
