# Reinforcement Learning By the Book

## Episode 1

### Problem Formulation

**Link:** [Reinforcement Learning By the Book (Ep. 1)](https://www.youtube.com/watch?v=NFo9v_yKQXA&list=PLzvYlJMoZ02Dxtwe-MmH4nOB5jYlMGBjr&index=1&pp=iAQB)

Reinforcement learning often makes certain assumptions about the agent and the environment. In this series, we will be using the Finite Markov Decision Process (MDP) framework to model the problem.

This framework describes the problem as follows:

- The agent is the decision maker / learner.
- The environment is the system that the agent interacts with.
- The agent's actions influence the environment and the agent's reward.
- The environment is fully observable, meaning the agent has access to the state of the environment at each time step.
- The future is Markovian, meaning the next state and reward only depend on the current state and action.
- The agent's goal is to learn a policy that maximizes the expected sum of future rewards.

At each time step the agent receives an observation of the environment's state ($s_t \in \mathcal{S}$), takes an action ($a_t \in \mathcal{A}$), and receives a reward ($r_{t+1} \in \mathcal{R} \subset \mathbb{R}$), along with a next state ($s_{t+1} \in \mathcal{S}$). The agent's goal is to learn a policy that maximizes the expected sum of future rewards.

The dynamics of the environment are given by the **MDP distribution function**, a probability function $p(s_{t+1}, r_{t+1} | s_t, a_t)$ which determines the probability of transitioning to state $s_{t+1}$ and receiving reward $r_{t+1}$ given that the agent is in state $s_t$ and takes action $a_t$.

The return $G_t$ is the sum of future rewards discounted by a factor $\gamma$:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots + \gamma^{T-t-1} r_T
$$

The discount factor $0 \leq \gamma \leq 1$ is a hyperparameter that controls the trade-off between immediate and future rewards. For example, if $\gamma = 0$, the agent is myopic and only cares about immediate rewards. If $\gamma = 1$, the agent is far-sighted and cares about future rewards as much as immediate rewards.

### Policy and Value Functions

The **policy** $\pi(a|s)$ is the probability of taking action $a$ in state $s$.

$$
\pi(a|s) = P[a_t = a | s_t = s]
$$

The **value function** $v_\pi(s)$ is the expected return starting from state $s$.

$$
v_\pi(s) = \mathbb{E}_\pi[G_t | s_t = s]
$$

The **action-value function** $q_\pi(s, a)$ is the expected return starting from state $s$ and taking action $a$.

$$
q_\pi(s, a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a]
$$

If the agent's policy is deterministic, then the value function and action-value function are equivalent:

$$
v_\pi(s) = q_\pi(s, \pi(s)) \quad \text{ if } \pi \text{ is deterministic}
$$

There is an optimal value function and action-value function for any MDP. Having access to the optimal action-value function allows the agent to make optimal decisions. The agent can simply select the action with the highest action-value for each state.

$$
v^*(s) = \max_\pi v_\pi(s) \quad \text{ and } \quad q^*(s, a) = \max_\pi q_\pi(s, a)
$$

However, finding the optimal value function and action-value function is intractable for most MDPs. Therefore, we need to approximate the optimal value function and action-value function.

## Episode 2 (The Bellman Equations & Policy Iteration)

**Link:** [Reinforcement Learning By the Book (Ep. 2)](https://www.youtube.com/watch?v=_j6pvGEchWU&list=PLzvYlJMoZ02Dxtwe-MmH4nOB5jYlMGBjr&index=2&pp=iAQB)

### Short Review

Finite Markov Decision Process (MDP) is a framework for modeling decision-making problems in reinforcement learning. An Agent takes actions in an environement which results in a reward and a next state. The agent's goal is to learn a policy that maximizes the expected sum of future rewards. In the finite case, there exists a set of special states called terminal states, which mark the end of the episode. Our goal is to find a policy (which is a probability distribution over actions for each state) that maximizes the expected sum of future rewards.

### Dynamic Programming

Algorithms used to find optimal policies which have complete information about the MDP.

### The Bellman Equations

The Bellman Equations are a set of recursive equations relating the value of a state or state-action pair to the value of its successor states or state-action pairs. It effectively allows us to break down the value of a state into the immediate reward plus the discounted value of the next state. If we knew the true value of the next state, we could compute the value of the current state, increasing the coverage of states we have accurate value estimates for.

For any policy $\pi$, we have the following Bellman Equation for the value function:

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} p(s', r | s, a) \left[ r + \gamma v_\pi(s') \right]
$$

And the following Bellman Equation for the action-value function:

$$
q_\pi(s, a) = \sum_{s' \in \mathcal{S}, r \in \mathcal{R}} p(s', r | s, a) \left[ r + \gamma \sum_{a' \in \mathcal{A}(s')} \pi(a' | s') q_\pi(s', a') \right]
$$

The Bellman Optimality Equations apply to the optimal value function and action-value function:

$$
v^*(s) = \max_{a \in \mathcal{A}(s)} \sum_{s' \in \mathcal{S}, r \in \mathcal{R}} p(s', r | s, a) \left[ r + \gamma v^*(s') \right]
$$

$$
q^*(s, a) = \sum_{s' \in \mathcal{S}, r \in \mathcal{R}} p(s', r | s, a) \left[ r + \gamma \max_{a'} q^*(s', a') \right]
$$

### Policy Evaluation

Compute either $v_\pi(s)$ or $q_\pi(s, a)$ for a given policy $\pi$. At this point, we will begin to work with estimates of the value function and action-value function, which we will denote as $V$ and $Q$ respectively. So the lower case $v$ and $q$ will be used to denote the true value function and action-value function, while the upper case $V$ and $Q$ will be used to denote the estimated value function and action-value function.

We start with an initial estimate of the value function or action-value function, and then iteratively update it using the Bellman Equations until it converges to the true value function or action-value function. Thanks to some nice properties of the Bellman Equations, we can guarantee convergence.

$$
V_\pi(s) \leftarrow \text{an initial estimate for } \pi
$$

Then our update rule is:

$$
V(s) \leftarrow \sum_{s' \in \mathcal{S}, r \in \mathcal{R}} p(s', r | s, a) \left[ r + \gamma V(s') \right]
$$

### Policy Improvement

Given an estimate of the value function, we can improve the policy by selecting the action that maximizes the action-value function for each state.

$$
\pi(s) \leftarrow \arg\max_a q(s, a)
$$

Thanks to the Policy Improvement Theorem, we know that the new policy will be as least as good as the old policy.

If $\pi$ is greedy with respect to $v_\pi$, then $v_\pi = v_*$ and $\pi = \pi_*$. So, if the policy improvement does not change the policy, then the policy must be optimal.

### Generalized Policy Iteration

Generalized Policy Iteration (GPI) is a framework for learning optimal policies. It involves two processes: policy evaluation and policy improvement.

From Sutton and Barto:

> [GPI refers] to the general idea of letting policy evaluation and policy improvement processes interact, independent of the granularity and other details of the two processes. Almost all RL methods are well described as GPI.

Policy evaluation moves the value function estimate ($V$) closer to the policy's true value function ($v_\pi$). Policy improvement replaces the policy ($\pi$) with a new policy that is greedy with respect to the current value function estimate ($\text{greedy}(V)$).

**Policy Iteration**: Policy evaluation to convergence followed by policy improvement.

$$
\pi_0 \xrightarrow{\text{E}} V_{\pi_0} \xrightarrow{\text{I}} \pi_1 \xrightarrow{\text{E}} V_{\pi_1} \xrightarrow{\text{I}} \pi_2 \xrightarrow{\text{E}} \cdots \xrightarrow{\text{I}} \pi_*
$$

Where $E$ stands for policy evaluation and $I$ stands for policy improvement. In Policy Iteration, we perform policy evaluation until convergence, then we perform policy improvement.

**Value Iteration**: Policy evaluation for a single step followed by policy improvement.

$$
\pi_0 \xrightarrow{\text{E}} V_{\pi_0} \xrightarrow{\text{I}} \pi_1 \xrightarrow{\text{E}} V_{\pi_1} \xrightarrow{\text{I}} \pi_2 \xrightarrow{\text{E}} \cdots \xrightarrow{\text{I}} \pi_*
$$

As long as both steps partially do their job, their interaction will drive the value function towards the optimal value function and the policy towards the optimal policy.

The naive implementations of Policy Iteration and Value Iteration are not scalable. They require a full sweep through the state space to evaluate the value function or action-value function for each policy update and they require full knowledge of the MDP dynamics.

## Episode 3 (Monte Carlo and Off-Policy Methods)

**Link:** [Reinforcement Learning By the Book (Ep. 3)](https://www.youtube.com/watch?v=bpUszPiWM7o&list=PLzvYlJMoZ02Dxtwe-MmH4nOB5jYlMGBjr&index=3&pp=iAQB)

### Monte Carlo (MC) Methods for RL

If we don't know the MDP dynamics, we can estimate the value function by simulating the environment. Monte Carlo methods are a class of algorithms that learn from episodes of experience.

MC uses:
- Averages to estimate expectations.
- GPI to obtain policies that are close to optimal.

MC does not estimate the MDP distribution function then proceed with GPI as though it is known.

### Model-Free vs Model-Based RL

A **Model** is a representation of the MDP dynamics. It is anything that the agent can use to predict the next state and reward given the current state and action.

- **Model-Free**: The agent does not attempt to learn a representation of the MDP dynamics. Instead, the agent simply learns correlations between actions and future rewards.
- **Model-Based**: The agent attempts to learn a representation of the MDP dynamics, then use planning to explore potential futures before making decisions.

MC is a model-free method. This means that it learns action-value functions directly from episodes of experience. We can't use the state value function because we need a model to connect actions to future states.

### Monte Carlo Evaluation

Goal: Given samples under a policy $\pi$, estimate $q_\pi(s, a)$ for all $s \in \mathcal{S}$ and $a \in \mathcal{A}(s)$.

We can express $q_\pi(s, a)$ estimation as $v_\pi(s)$ estimation. That's because we can interpret actions as part of the environment, and study a **Markov Reward Process (MRP)** instead of an MDP. This is an MDP without actions. If each state is a tuple of (state, action) from the original MDP, then the dynamics of the MRP are the same as the dynamics of the MDP.

### Constant-Alpha Monte Carlo

**Update Rule:**

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ G_t - Q(s, a) \right]
$$

Where $G_t$ is the return following time step $t$ and $\alpha$ is the learning rate.

This allows us to update our estimate of the action-value function for each state-action pair without having to know the MDP dynamics but it requires many episodes to converge and it assumes the policy is fixed.

### Exploration-Exploitation Trade-off

"To discover the optimal policy, the agent must explore all actions in all states, at least once. **BUT** to get high returns, the agent must exploit its knowledge of high value state-action pairs in the environment."

With infinite data, the optimal policy is discoverable as long as the policy is soft (i.e. all actions have a non-zero probability of being taken). An example of a soft policy is $\epsilon$-greedy. This policy balances exploration and exploitation by selecting actions at random with probability $\epsilon$ and selecting the greedy action with probability $1 - \epsilon$.

### Constant-$\alpha$ MC for Estimating $\pi$ in Blackjack

**State:** (player's sum, dealer's show card, usable ace)

**Actions:** (hit, stick)

**Hyperparameters:**
- $\alpha \in (0, 1)$ (i.e. $1/5000$)
- $\epsilon \in (0, 1)$ (i.e. $0.1$)
- $M$ episodes (i.e. $10 \text{ million}$)

**Initialize:**
- $Q(s, a) \leftarrow \text{Any } \epsilon\text{-greedy policy}$

**Algorithm:**

For each episode, generate an episode following $\pi$ using the current $Q$.

For each timestep $t$ in the episode,

1. Compute the return:

$$
G_t \leftarrow r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{T-1} r_T
$$

2. Update the action-value function:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ G_t - Q(s_t, a_t) \right]
$$

Then update the policy to be the $\epsilon$-greedy policy with respect to the current action-value function estimate:

$$
\pi(s) \leftarrow \epsilon\text{-greedy}(Q)
$$

More explicitly, the new policy is:

$$
\pi(s) = \begin{cases} 
1 - \epsilon + \frac{\epsilon}{|\mathcal{A}(s)|}
&
\text{if } a = \arg\max_a Q(s, a)
\\
\frac{\epsilon}{|\mathcal{A}(s)|}
&
\text{otherwise}
\end{cases}
$$

### Off-Policy Methods

Off-policy methods are a generalization of on-policy methods. They learn from a policy different from the one being followed / optimized.

In the previous example, we use the policy for two things: generating the data/episodes and updating the policy. Off-policy methods separate these two tasks. The policy used to generate the data is called the **behavior policy** ($\beta(a|s)$) and the policy being learned is called the **target policy** ($\pi(a|s)$).

In on-policy methods, the behavior policy is the same as the target policy.

$$
\beta = \pi \quad \text{for on-policy methods}
$$

$$
\beta \neq \pi \quad \text{for off-policy methods}
$$

Since we are observing data generated by the behavior policy, directly estimating $q_\pi(s, a)$ takes an expectation with respect to the behavior policy.

$$
q_\pi(s, a) = \mathbb{E}_{\beta}[G_t | S_t = s, A_t = a]
$$

But we want to estimate this expectation with respect to the target policy. We can do this by weighting the return by the ratio of the target policy probability to the behavior policy probability. This is called **importance sampling**.

$$
q_\pi(s, a) = \mathbb{E}_{\beta}[ \frac{p_\pi(G_t)}{p_\beta(G_t)} G_t | S_t = s, A_t = a]
$$


The **importance sampling ratio** is the ratio of the target policy probability to the behavior policy probability.
$$
\rho_{t} = \frac{p_\pi(G_t)}{p_\beta(G_t)} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}
$$

The **off-policy correction** is the importance sampling ratio multiplied by the return.

$$
\hat{q}(s_t, a_t) \leftarrow \hat{q}(s_t, a_t) + \alpha \rho_{t} \left[ G_t - \hat{q}(s_t, a_t) \right]
$$