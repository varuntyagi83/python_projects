import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
from scipy.stats import chi2_contingency, beta
from datetime import datetime, timedelta
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import random

def generate_synthetic_data(n=20000):
    np.random.seed(42)
    user_ids = np.arange(n)
    created_dates = [datetime.now() - timedelta(days=np.random.randint(0, 1000)) for _ in range(n)]
    age_groups = np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], size=n)
    install_dates = [created_date + timedelta(days=np.random.randint(0, 30)) for created_date in created_dates]
    channels = np.random.choice(['Organic', 'Paid', 'Referral'], size=n)
    game_types = np.random.choice(['Puzzle', 'Strategy', 'Arcade'], size=n)
    locations = np.random.choice(['US', 'UK', 'CA', 'AU', 'IN'], size=n)
    last_activity_dates = [install_date + timedelta(days=np.random.randint(0, 300)) for install_date in install_dates]
    cohort_ages = [(datetime.now() - created_date).days for created_date in created_dates]
    groups = np.random.choice(['A', 'B'], size=n)

    age_group_numeric = (age_groups == '25-34').astype(float)
    channel_numeric = (channels == 'Paid').astype(float)
    game_type_numeric = (game_types == 'Puzzle').astype(float)
    location_numeric = (locations == 'US').astype(float)

    base_log_odds_A = -2.0
    base_log_odds_B = 2.0

    coef_age = 1.5
    coef_channel = 1.8
    coef_game = 1.2
    coef_location = 1.0
    coef_interaction = 2.0

    log_odds_A = (base_log_odds_A + 
                  coef_age * age_group_numeric + 
                  coef_channel * channel_numeric + 
                  coef_game * game_type_numeric + 
                  coef_location * location_numeric + 
                  coef_interaction * age_group_numeric * channel_numeric +
                  np.random.normal(0, 0.3, n))

    log_odds_B = (base_log_odds_B + 
                  coef_age * age_group_numeric + 
                  coef_channel * channel_numeric + 
                  coef_game * game_type_numeric + 
                  coef_location * location_numeric + 
                  coef_interaction * age_group_numeric * channel_numeric +
                  np.random.normal(0, 0.3, n))

    probs_A = 1 / (1 + np.exp(-log_odds_A))
    probs_B = 1 / (1 + np.exp(-log_odds_B))

    conversions = np.where(
        groups == 'A',
        np.random.binomial(1, probs_A),
        np.random.binomial(1, probs_B)
    )

    return pd.DataFrame({
        'user_id': user_ids,
        'created_date': created_dates,
        'age_group': age_groups,
        'install_date': install_dates,
        'channel': channels,
        'game_type': game_types,
        'location': locations,
        'last_activity_date': last_activity_dates,
        'cohort_age': cohort_ages,
        'group': groups,
        'converted': conversions
    })

def preprocess_data(data):
    data = data.copy()
    categorical_columns = ['age_group', 'channel', 'game_type', 'location', 'group']
    for col in categorical_columns:
        data[col] = data[col].astype('category').cat.codes
    return data

def evaluate_model(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    chi2, p_value, _, _ = chi2_contingency(cm)
    return cm, accuracy, precision, recall, f1, p_value

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def bayesian_ab_test(data_a, data_b, n_simulations=100000):
    a_success = np.sum(data_a)
    a_trials = len(data_a)
    b_success = np.sum(data_b)
    b_trials = len(data_b)
    
    a_posterior = beta(a_success + 1, a_trials - a_success + 1)
    b_posterior = beta(b_success + 1, b_trials - b_success + 1)
    
    a_samples = a_posterior.rvs(n_simulations)
    b_samples = b_posterior.rvs(n_simulations)
    
    prob_b_better = np.mean(b_samples > a_samples)
    expected_loss = np.mean(np.maximum(a_samples - b_samples, 0))
    
    return prob_b_better, expected_loss

def plot_posterior_distributions(data_a, data_b):
    a_success = np.sum(data_a)
    a_trials = len(data_a)
    b_success = np.sum(data_b)
    b_trials = len(data_b)
    
    a_posterior = beta(a_success + 1, a_trials - a_success + 1)
    b_posterior = beta(b_success + 1, b_trials - b_success + 1)
    
    x = np.linspace(0, 1, 1000)
    plt.figure(figsize=(10, 6))
    plt.plot(x, a_posterior.pdf(x), label='A')
    plt.plot(x, b_posterior.pdf(x), label='B')
    plt.xlabel('Conversion Rate')
    plt.ylabel('Density')
    plt.title('Posterior Distributions')
    plt.legend()
    plt.show()

def multi_armed_bandit(data, n_rounds=1000, alpha=0.05):
    arms = ['A', 'B']
    counts = {arm: 0 for arm in arms}
    rewards = {arm: 0 for arm in arms}

    for _ in range(n_rounds):
        arm = random.choice(arms)
        counts[arm] += 1
        if arm == 'A':
            reward = np.random.binomial(1, data[data['group'] == 'A']['converted'].mean())
        else:
            reward = np.random.binomial(1, data[data['group'] == 'B']['converted'].mean())
        rewards[arm] += reward

    # Calculate statistical significance (p-value)
    _, p_value = stats.ttest_ind(
        data[data['group'] == 'A']['converted'],
        data[data['group'] == 'B']['converted']
    )

    significant = p_value < alpha
    return rewards, p_value, significant

# Reinforcement Learning (Simplified version)
def reinforcement_learning(data, alpha=0.05, n_iterations=1000):
    np.random.seed(42)
    total_reward = 0
    rewards_A = []
    rewards_B = []
    from scipy import stats # import stats module

    # Ensure you are referencing the column in the dataframe using the correct values 'A' and 'B'
    mean_conversion_A = data[data['group'] == 'A']['converted'].mean() 
    mean_conversion_B = data[data['group'] == 'B']['converted'].mean()

    for _ in range(n_iterations):
        action = np.random.choice([0, 1])
        if action == 0:
            reward = np.random.binomial(1, mean_conversion_A)
            rewards_A.append(reward)
        else:
            reward = np.random.binomial(1, mean_conversion_B)
            rewards_B.append(reward)
        total_reward += reward

    # Use Mann-Whitney U test
    _, p_value = stats.mannwhitneyu(rewards_A, rewards_B, alternative='two-sided')

    significant = p_value < alpha
    avg_reward = total_reward / n_iterations

    # Return dummy values for cm, accuracy, precision, recall, and f1
    cm = None 
    accuracy = None
    precision = None
    recall = None
    f1 = None
    
    return cm, accuracy, precision, recall, f1, avg_reward, p_value, significant

# Generate and preprocess data
data = generate_synthetic_data()
preprocessed_data = preprocess_data(data)

# Prepare features and target
feature_columns = ['age_group', 'channel', 'game_type', 'location', 'cohort_age', 'group']
X = preprocessed_data[feature_columns]
y = preprocessed_data['converted']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "SVM": SVC(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    "Reinforcement Learning": reinforcement_learning
}

# Train and evaluate models
for model_name, model in models.items():
    if model_name == "Reinforcement Learning":
        cm, accuracy, precision, recall, f1, avg_reward, p_value, significant = model_func(data)
        
        # Print Reinforcement Learning Results
        print(f"\n{model_name} Results:")
        print(f"Average Reward: {avg_reward:.4f}")
        print(f"P-value: {p_value:.4e}")
        print(f"Statistically Significant: {significant}") 
        
    else:    
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm, accuracy, precision, recall, f1, p_value = evaluate_model(y_test, y_pred)
        
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"P-value: {p_value:.4e}")
        print(f"Statistically Significant: {p_value < 0.05}")
        print("Confusion Matrix:")
        print(cm)
        
        plot_confusion_matrix(cm, model_name)

# Bayesian A/B Test
data_a = data[data['group'] == 'A']['converted'].values
data_b = data[data['group'] == 'B']['converted'].values

prob_b_better, expected_loss = bayesian_ab_test(data_a, data_b)

print("\nBayesian A/B Test Results:")
print(f"Probability that B is better than A: {prob_b_better:.4f}")
print(f"Expected loss of choosing A over B: {expected_loss:.4f}")

plot_posterior_distributions(data_a, data_b)

# Multi-Armed Bandit
rewards, p_value, significant = multi_armed_bandit(data)

print("\nMulti-Armed Bandit Results:")
print(f"Rewards for A: {rewards['A']}")
print(f"Rewards for B: {rewards['B']}")
print(f"P-value: {p_value:.4f}")
print(f"Statistically Significant: {significant}")

# Plot Multi-Armed Bandit results
plt.figure(figsize=(10, 6))
plt.bar(rewards.keys(), rewards.values())
plt.title("Multi-Armed Bandit Rewards")
plt.xlabel("Arm")
plt.ylabel("Total Reward")
plt.show()

# Print conversion rates for groups A and B
conversion_rate_A = data[data['group'] == 'A']['converted'].mean()
conversion_rate_B = data[data['group'] == 'B']['converted'].mean()
print(f"\nConversion Rate A: {conversion_rate_A:.4f}")
print(f"Conversion Rate B: {conversion_rate_B:.4f}")
