# ----------------------------
# SECTION 1: Data Preprocessing
# ----------------------------
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pandas as pd
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from Google Drive
from google.colab import files, drive

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

drive.mount('/content/drive')
data = pd.read_csv('/content/drive/My Drive/Data/bnpl_credit_data_200.csv')

# Preprocess
data.fillna(0, inplace=True)
data['Age Condition'] = np.where(data['Age'] < 18, 0, 1)
data['Credit_Condition'] = np.where(data['Credit Score'] > 519, 1, 0)

freq_cols = [f'Monthly Purchase Frequency {i}' for i in range(1, 7)]
amount_cols = [f'Monthly Purchase Amount {i}' for i in range(1, 7)]
data['Total_Purchase_Frequency'] = data[freq_cols].sum(axis=1)
data['Total_Purchase_Amount'] = data[amount_cols].sum(axis=1)
data['Repeat Usage'] = data['Repeat Usage'].map({'Yes': 1, 'No': 0})

# Credit logic
def determine_credit(row):
    if row['Credit_Condition'] == 0:
        return 0, 0  # No credit
    if row['Payment Status'] == 'No':
        if row['Total_Purchase_Amount'] > 310000001:
            return 10000000, 1  # 10M for 1 month
        elif row['Total_Purchase_Amount'] > 150000001:
            return 5000000, 1  # 5M for 1 month
    else:
        if row['Total_Purchase_Frequency'] > 79 and row['Total_Purchase_Amount'] > 220000000:
            return 10000000, 3  # 10M for 3 months
        elif row['Total_Purchase_Frequency'] > 79 and row['Total_Purchase_Amount'] < 220000001:
            return 10000000, 1  # 10M for 1 month
        elif row['Total_Purchase_Frequency'] < 80 and row['Total_Purchase_Amount'] > 110000000:
            return 5000000, 3  # 5M for 3 months
        elif row['Total_Purchase_Frequency'] < 80 and row['Total_Purchase_Amount'] < 1100000001:
            return 5000000, 1  # 5M for 1 month
        elif row['Total_Purchase_Frequency'] < 41 and row['Total_Purchase_Amount'] < 80000001:
            return 2000000, 1  # 2M for 1 month
    return 0, 0  # Default no credit


data[['Credit Amount', 'Repayment Period']] = data.apply(determine_credit, axis=1, result_type='expand')
data['Target'] = np.where(data['Credit_Condition'] & (data['Total_Purchase_Amount'] > 10), 1, 0)

features = data[['Age', 'Credit Score', 'Total_Purchase_Frequency', 'Total_Purchase_Amount', 'Age Condition', 'Rating', 'Repeat Usage']]
target = data['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.3, random_state=42)


# ----------------------------
# SECTION 2: Discrete SAC Agent
# ----------------------------
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        return probs, log_probs

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(tuple(args))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*samples))

    def __len__(self):
        return len(self.buffer)

class SimpleBNPLEnv:
    def __init__(self, X, y, action_space):
        self.X = X
        self.y = y
        self.index = 0
        self.action_space = action_space
        self.n = len(X)

    def reset(self):
        self.index = 0
        return self.X[self.index]

    def step(self, action):
        credit_amount, months = self.action_space[action]
        y_true = self.y[self.index]
        if y_true == 1:
            reward = 2 if credit_amount > 0 else -2
        else:
            reward = -1 if credit_amount > 0 else 1
        self.index += 1
        done = self.index >= self.n
        next_state = self.X[self.index] if not done else self.X[0]
        return next_state, reward, done

class DiscreteSACAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, alpha=0.2, tau=0.005):
        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim).to(device)
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.q1_optim = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=3e-4)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=3e-4)

        self.memory = ReplayBuffer(100000)
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.action_dim = action_dim

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs, _ = self.policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        with torch.no_grad():
            next_probs, next_log_probs = self.policy(next_states)
            q1_next = self.q1_target(next_states)
            q2_next = self.q2_target(next_states)
            min_q_next = torch.min(q1_next, q2_next)
            next_value = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=1)
            target_q = rewards + self.gamma * (1 - dones) * next_value

        q1_vals = self.q1(states).gather(1, actions.unsqueeze(1)).squeeze()
        q2_vals = self.q2(states).gather(1, actions.unsqueeze(1)).squeeze()

        q1_loss = F.mse_loss(q1_vals, target_q)
        q2_loss = F.mse_loss(q2_vals, target_q)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        probs, log_probs = self.policy(states)
        min_q = torch.min(self.q1(states), self.q2(states))
        policy_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

# ----------------------------
# SECTION 2.5: Discrete SAC Training Loop
# ----------------------------
# Setup environment and agent
state_dim = X_train.shape[1]
action_space = [(0, 0), (2000000, 1), (5000000, 1), (5000000, 3), (10000000, 1), (10000000, 3)]
action_dim = len(action_space)

agent = DiscreteSACAgent(state_dim=state_dim, action_dim=action_dim)
env = SimpleBNPLEnv(X_train, y_train.values, action_space)

num_episodes = 50
batch_size = 64
rewards_per_episode = []
train_accuracies, val_accuracies, train_losses = [], [], []

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    losses = []

    for t in range(env.n):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.memory.push(state, action, reward, next_state, done)
        agent.update(batch_size)
        state = next_state
        episode_reward += reward
        if done:
            break

    rewards_per_episode.append(episode_reward)

    # Accuracy evaluation
    with torch.no_grad():
        X_train_tensor_eval = torch.tensor(X_train, dtype=torch.float32)
        probs_train, _ = agent.policy(X_train_tensor_eval.to(device))
        pred_train = torch.argmax(probs_train, dim=1).cpu().numpy()
        train_accuracy = accuracy_score(y_train, pred_train != 0)

        X_test_tensor_eval = torch.tensor(X_test, dtype=torch.float32)
        probs_test, _ = agent.policy(X_test_tensor_eval.to(device))
        pred_test = torch.argmax(probs_test, dim=1).cpu().numpy()
        val_accuracy = accuracy_score(y_test, pred_test != 0)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    train_losses.append(0.0)  # Discrete SAC loss چندگانه است، اگر خواستی می‌تونیم میانگینش رو ذخیره کنیم

    print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")


# Show total reward per episode
import matplotlib.pyplot as plt
plt.plot(rewards_per_episode)
plt.title('Total Reward per Episode (SAC)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()



# ----------------------------
# SECTION 3: SAC Evaluation Metrics
# ----------------------------
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score

# Predict with SAC policy network
agent.policy.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    probs, _ = agent.policy(X_test_tensor)
    best_actions = torch.argmax(probs, dim=1).cpu().numpy()

# If action != 0 → credit offered
predictions = (best_actions != 0).astype(int)
y_true = y_test.values

# Metrics
accuracy = accuracy_score(y_true, predictions)
recall = recall_score(y_true, predictions)
precision = precision_score(y_true, predictions)
f1 = f1_score(y_true, predictions)
auc_score_val = roc_auc_score(y_true, predictions)
conf_matrix = confusion_matrix(y_true, predictions)

# KS Statistic
results_df = pd.DataFrame({'Actual': y_true, 'Predicted': predictions})
results_df['Positive'] = np.where(results_df['Actual'] == 1, results_df['Predicted'], 0)
results_df['Negative'] = np.where(results_df['Actual'] == 0, results_df['Predicted'], 0)
cum_pos = results_df['Positive'].cumsum() / results_df['Positive'].sum()
cum_neg = results_df['Negative'].cumsum() / results_df['Negative'].sum()
ks_statistic = np.max(np.abs(cum_pos - cum_neg))

# Print metrics
print("--- SAC Evaluation Metrics ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc_score_val:.4f}")
print(f"KS Statistic: {ks_statistic:.4f}")
print("Confusion Matrix:", conf_matrix)


# ----------------------------
# SECTION 4: DQN Evaluation Plots
# ----------------------------
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import shap

# KS Curve
plt.figure(figsize=(10, 6))
plt.plot(cum_pos, label='Cumulative Positive', color='blue')
plt.plot(cum_neg, label='Cumulative Negative', color='orange')
plt.axhline(y=ks_statistic, color='red', linestyle='--', label='KS Statistic')
plt.title('KS Statistic Curve')
plt.xlabel('Threshold')
plt.ylabel('Cumulative Distribution')
plt.legend()
plt.grid(True)
plt.show()

# Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Precision-Recall Curve
precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_true, predictions)
plt.figure(figsize=(8, 5))
plt.plot(recall_vals, precision_vals, marker='.', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.legend()
plt.show()

# ROC Curve
fpr, tpr, thresholds_roc = roc_curve(y_true, predictions)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)
plt.legend()
plt.show()

# Accuracy Plot
plt.figure(figsize=(8, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Loss Plot
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Scatter Accuracy
metrics_df = pd.DataFrame({'Accuracy': train_accuracies, 'Val_Accuracy': val_accuracies})
plt.figure(figsize=(8, 6))
sns.scatterplot(data=metrics_df, x='Accuracy', y='Val_Accuracy')
plt.title('Train vs Validation Accuracy')
plt.grid(True)
plt.show()

# Box Plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Payment Status', y='Total_Purchase_Amount', data=data)
plt.xlabel('Payment Status')
plt.ylabel('Total Purchase Amount')
plt.title('Box Plot by Payment Status')
plt.show()

# SHAP Analysis
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Linear(X_train.shape[1], 1)

    def forward(self, x):
        return self.layer(x)

shap_model = SimpleModel()
def model_predict(x):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    if x_tensor.ndim == 1:
        x_tensor = x_tensor.unsqueeze(0)
    with torch.no_grad():
        return shap_model(x_tensor).detach().numpy()

explainer = shap.Explainer(model_predict, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, feature_names=features.columns.tolist())


# Save the results to a new CSV file
data.to_csv('customer_credit_offers_SAC.csv', index=False)
files.download('customer_credit_offers_SAC.csv')

