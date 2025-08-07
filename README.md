# Discrete SAC Agent for BNPL Credit Strategy 💳

This project implements a Discrete Soft Actor-Critic (SAC) agent to optimize credit recommendations in a Buy Now, Pay Later (BNPL) system using reinforcement learning.

---

## 🎯 Objective

Train a deep reinforcement learning agent that learns which credit offer to give to a user (or to reject) based on their features and historical behavior.

---

## 🧠 Model Components

### ▪ QNetwork
Approximates Q-values for each state-action pair. Two Q-networks (`q1`, `q2`) are used to reduce overestimation.

### ▪ PolicyNetwork
A softmax-based policy that outputs a distribution over actions. Used to sample actions and compute entropy-regularized policy loss.

### ▪ ReplayBuffer
Stores experiences `(state, action, reward, next_state, done)` for off-policy training.

### ▪ SimpleBNPLEnv
Simulates user credit behavior with defined rewards:
- ✅ Credit + Pay → +2
- ❌ Credit + NoPay → -1
- ❌ NoCredit + Pay → -2
- ✅ NoCredit + NoPay → +1

---

## ⚙️ Agent Training

- Entropy-regularized updates using Soft Actor-Critic formulation
- Target Q-networks updated with soft updates (τ = 0.005)
- Optimizers: Adam with learning rate 3e-4
- γ (discount) = 0.99, α (entropy weight) = 0.2

---

## 📊 Metrics Tracked

- 🔹 Reward per episode
- 🔹 Training accuracy (credit decision vs true label)
- 🔹 Validation accuracy

---

## 📂 Files

- `sac_model.py`: All models and agent
- `train.py`: Training loop
- `README.md`: Project documentation

---

## 📦 Requirements

```bash
torch
numpy
scikit-learn
