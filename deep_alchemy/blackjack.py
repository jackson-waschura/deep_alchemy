"""
Blackjack example from Sutton and Barto.

This script implements a blackjack environment and a Monte Carlo agent that learns the optimal policy for the game.

It first uses numpy to implement the environment and agent.

Then it reimplements them using Jax.

Finally, it also implements both on-policy and off-policy Monte Carlo methods.
"""
import argparse
import numpy as np
from tqdm import tqdm
from enum import Enum
from dataclasses import dataclass
import time
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Blackjack example")
    parser.add_argument("--alpha", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--M", type=int, default=1_000_000, help="Number of episodes to run")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
    parser.add_argument("--show_progress", action="store_true", help="Show progress of training")
    parser.add_argument("--play", action="store_true", help="Play blackjack in interactive mode")
    # TODO: Add argument for target policy initialization
    # TODO: Add argument for behavior policy
    return parser.parse_args()

class Action(Enum):
    HIT = 0
    STICK = 1

@dataclass(frozen=True)
class State:
    player_sum: int
    dealer_show: int
    usable_ace: bool

@dataclass
class Timestep:
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool

class BlackjackEnvironment:
    def __init__(self):
        # Use uint8 to save memory since cards are small integers
        self.fresh_deck = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4, dtype=np.uint8)
        self.deck_idx = 0
        
    def _draw_card(self) -> int:
        """Draw a card from the deck"""
        if self.deck_idx >= len(self.fresh_deck):
            # Reshuffle if we run out of cards
            np.random.shuffle(self.deck)
            self.deck_idx = 0
        card = self.deck[self.deck_idx].item()
        self.deck_idx += 1
        return card

    def _compute_hand_value(self, cards: list[int]) -> tuple[int, bool]:
        """
        Compute the value of a hand and whether it has a usable ace.
        Returns (hand_value, usable_ace)
        """
        value = sum(cards)
        num_aces = cards.count(11)
        
        # Check for aces
        if num_aces > 0:
            # Keep reducing aces by 10 until we're under 21
            while value > 21 and num_aces > 0:
                value -= 10
                num_aces -= 1
                
        return value, num_aces > 0

    def reset(self) -> State:
        """Reset the environment and sample an initial state."""
        self.deck = self.fresh_deck.copy()
        np.random.shuffle(self.deck)
        self.deck_idx = 0
            
        # Deal initial cards
        self.player_cards = [self._draw_card(), self._draw_card()]
        self.dealer_cards = [self._draw_card(), self._draw_card()]

        # Draw up to a sum of 12
        while sum(self.player_cards) < 12:
            self.player_cards.append(self._draw_card())
        
        # Compute initial state
        player_sum, usable_ace = self._compute_hand_value(self.player_cards)
        return State(player_sum=player_sum, 
                    dealer_show=self.dealer_cards[0], 
                    usable_ace=usable_ace)

    def step(self, action: Action) -> tuple[State, float, bool]:
        """Take a step given an action. Returns (next_state, reward, done)."""
        if action == Action.HIT:
            # Player hits
            self.player_cards.append(self._draw_card())
            player_sum, usable_ace = self._compute_hand_value(self.player_cards)
            
            # Check if player busts
            if player_sum > 21:
                return State(player_sum=player_sum,
                           dealer_show=self.dealer_cards[0],
                           usable_ace=usable_ace), -1.0, True
            elif player_sum < 21:
                # Game continues
                return State(player_sum=player_sum,
                           dealer_show=self.dealer_cards[0],
                           usable_ace=usable_ace), 0.0, False
        # Dealer's turn
        dealer_sum, _ = self._compute_hand_value(self.dealer_cards)
        
        # Dealer hits on 16 or below, stands on 17 or above
        while dealer_sum < 17:
            self.dealer_cards.append(self._draw_card())
            dealer_sum, _ = self._compute_hand_value(self.dealer_cards)
        
        # Compare hands
        player_sum, usable_ace = self._compute_hand_value(self.player_cards)
        
        # Dealer busts
        if dealer_sum > 21:
            return State(player_sum=player_sum,
                        dealer_show=self.dealer_cards[0],
                        usable_ace=usable_ace), 1.0, True
                        
        # Compare final hands
        if player_sum > dealer_sum:
            reward = 1.0
        elif player_sum < dealer_sum:
            reward = -1.0
        else:
            reward = 0.0  # Push
            
        return State(player_sum=player_sum,
                    dealer_show=self.dealer_cards[0],
                    usable_ace=usable_ace), reward, True

class BlackjackMonteCarloAgent:
    def __init__(self, alpha: float, gamma: float, epsilon: float):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = self.initialize_Q()

    def initialize_Q(self) -> dict[State, dict[Action, float]]:
        """
        Initialize the Q function to be 0 for all state-action pairs.
        """
        Q = {}
        for player_sum in range(12, 22):
            for dealer_show in range(2, 12):
                for usable_ace in [True, False]:
                    Q[State(player_sum, dealer_show, usable_ace)] = {Action.HIT: 0.0, Action.STICK: 0.0}
        return Q

    def take_action(self, state: State) -> Action:
        """
        Take an action according to the epsilon-greedy policy.
        """
        policy = self.Q[state]
        if np.random.random() < self.epsilon:
            return np.random.choice(list(policy.keys()))
        else:
            return max(policy, key=policy.get)

    def take_action_eval(self, state: State) -> Action:
        """
        Take an action according to the greedy policy.
        """
        policy = self.Q[state]
        return max(policy, key=policy.get)

    def generate_episode(self, env: BlackjackEnvironment, training: bool = True) -> list[Timestep]:
        """
        Generate an episode using the given policy.
        """
        state = env.reset()
        done = False
        episode = []
        while not done:
            action = self.take_action(state) if training else self.take_action_eval(state)
            next_state, reward, done = env.step(action)
            episode.append(Timestep(state, action, reward, next_state, done))
            state = next_state
        return episode

    def update(self, episode: list[Timestep]):
        """
        Update the agent's Q function using the given episode.
        """
        g_t = 0.0
        for timestep in reversed(episode):
            g_t = self.gamma * g_t + timestep.reward
            self.Q[timestep.state][timestep.action] += self.alpha * (g_t - self.Q[timestep.state][timestep.action])

def train_agent(agent: BlackjackMonteCarloAgent, env: BlackjackEnvironment, num_episodes: int, show_progress: bool):
    """
    Train the agent for the given number of episodes.
    Returns the trained agent.
    """
    iterator = range(num_episodes)
    if show_progress:
        iterator = tqdm(iterator, desc="Training", smoothing=0.001, mininterval=0.2)  # Update at most every 0.2 seconds
    ema_reward = None
    last_print_t = time.time()
    for _ in iterator:
        episode = agent.generate_episode(env)
        ema_reward = episode[-1].reward if ema_reward is None else 0.999 * ema_reward + 0.001 * episode[-1].reward
        agent.update(episode)
        if show_progress:
            if last_print_t + 0.2 < time.time():
                iterator.set_postfix(reward=f"{ema_reward:7.3f}")
                last_print_t = time.time()
            
    return agent

def train_agent_notebook(
    agent: BlackjackMonteCarloAgent, 
    env: BlackjackEnvironment, 
    num_episodes: int = 1_000_000, 
    eval_episodes: int = 100_000,
    eval_interval: int = 10_000,
    show_progress: bool = False
):
    """
    Train the agent in a notebook environment and plot training statistics.
    Evaluates agent performance periodically during training using a separate evaluation loop.
    
    Args:
        agent: The BlackjackMonteCarloAgent to train
        env: The BlackjackEnvironment to use
        num_episodes: Total number of training episodes
        eval_episodes: Number of episodes to run for each evaluation
        eval_interval: Number of training episodes between evaluations
        show_progress: Whether to show a progress bar
    """
    # Calculate number of evaluation points
    num_evals = (num_episodes // eval_interval) + 1  # +1 for initial evaluation
    
    # Pre-allocate arrays for statistics
    mean_rewards = np.zeros(num_evals)
    std_rewards = np.zeros(num_evals)
    q_value_changes = np.zeros(num_evals)  # New array for Q-value changes
    eval_points = np.zeros(num_evals)
    
    def evaluate_agent():
        """Run evaluation episodes and return statistics."""
        rewards = []
        for _ in range(eval_episodes):
            episode = agent.generate_episode(env, training=False)
            rewards.append(episode[-1].reward)
        return np.mean(rewards), np.std(rewards)
    
    def compute_q_values_snapshot():
        """Create a snapshot of current Q-values."""
        snapshot = []
        for state in agent.Q:
            for action in agent.Q[state]:
                snapshot.append(agent.Q[state][action])
        return np.array(snapshot)
    
    # Initial evaluation and Q-value snapshot
    mean_rewards[0], std_rewards[0] = evaluate_agent()
    eval_points[0] = 0
    previous_q_values = compute_q_values_snapshot()
    q_value_changes[0] = 0  # No change for first evaluation

    iterator = range(num_episodes)
    if show_progress:
        iterator = tqdm(iterator, desc="Training", smoothing=0.001, mininterval=0.2)
    
    # Training loop with periodic evaluation
    eval_idx = 1
    for i in iterator:
        # Training step
        episode = agent.generate_episode(env, training=True)
        agent.update(episode)
        
        # Evaluate periodically
        if (i + 1) % eval_interval == 0:
            mean_rewards[eval_idx], std_rewards[eval_idx] = evaluate_agent()
            eval_points[eval_idx] = i + 1
            
            # Compute Q-value changes
            current_q_values = compute_q_values_snapshot()
            q_value_changes[eval_idx] = np.mean(np.abs(current_q_values - previous_q_values))
            previous_q_values = current_q_values
            
            if show_progress:
                iterator.set_postfix(
                    eval_reward=f"{mean_rewards[eval_idx-1]:7.3f}",
                    q_change=f"{q_value_changes[eval_idx]:7.5f}"
                )
            
            eval_idx += 1
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[1, 1])
    
    # Plot rewards
    ax1.plot(eval_points, mean_rewards, label='Mean Reward', color='blue')
    ax1.fill_between(eval_points, 
                     mean_rewards - std_rewards,
                     mean_rewards + std_rewards,
                     alpha=0.2, color='blue', label='±1 std dev')
    
    ax1.set_xlabel('Training Episodes')
    ax1.set_ylabel('Evaluation Reward')
    ax1.set_title(f'Blackjack Agent Evaluation (every {eval_interval} episodes)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Q-value changes
    ax2.plot(eval_points[1:], q_value_changes[1:], color='green', label='Q-value Change')
    ax2.set_xlabel('Training Episodes')
    ax2.set_ylabel('Mean Absolute Q-value Change')
    ax2.set_title('Q-value Stability')
    ax2.set_yscale('log')  # Use log scale for better visualization
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def display_hand(cards: list[int], hide_second: bool = False) -> str:
    """Format a hand for display, optionally hiding the second card."""
    if hide_second and len(cards) >= 2:
        return f"{cards[0]}, XX"
    return ", ".join(str(card) for card in cards)

def play_blackjack(env: BlackjackEnvironment):
    """Interactive CLI for playing blackjack."""
    print("\nWelcome to Blackjack!")
    print("===================")
    
    while True:
        # Reset the environment
        state = env.reset()
        
        while True:
            # Clear some space
            print("\n" + "="*40 + "\n")
            
            # Show current game state
            print(f"Dealer's cards: {display_hand(env.dealer_cards, hide_second=True)}")
            print(f"Your cards: {display_hand(env.player_cards)} (Total: {state.player_sum})")
            
            if state.player_sum >= 21:
                break
                
            # Get player action
            while True:
                action_input = input("\nWhat would you like to do? [H]it or [S]tick? ").lower()
                if action_input in ['h', 's']:
                    break
                print("Invalid input! Please enter 'h' for Hit or 's' for Stick.")
            
            # Convert input to action
            action = Action.HIT if action_input == 'h' else Action.STICK
            
            # Take step in environment
            state, reward, done = env.step(action)
            
            if done:
                break
        
        # Show final hands
        print("\nGame Over!")
        print(f"Dealer's cards: {display_hand(env.dealer_cards)} (Total: {sum(env.dealer_cards)})")
        print(f"Your cards: {display_hand(env.player_cards)} (Total: {state.player_sum})")
        
        # Show result
        if reward > 0:
            print("\nYou win! 🎉")
        elif reward < 0:
            print("\nYou lose! 😢")
        else:
            print("\nIt's a tie! 🤝")
            
        # Ask to play again
        while True:
            play_again = input("\nWould you like to play again? [Y/N] ").lower()
            if play_again in ['y', 'n']:
                break
            print("Invalid input! Please enter 'Y' for Yes or 'N' for No.")
            
        if play_again == 'n':
            break
    
    print("\nThanks for playing!")


if __name__ == "__main__":
    args = parse_args()
    
    env = BlackjackEnvironment()
    
    if args.play:
        play_blackjack(env)
    else:
        agent = BlackjackMonteCarloAgent(args.alpha, args.gamma, args.epsilon)
        agent = train_agent(agent, env, args.M, args.show_progress)
