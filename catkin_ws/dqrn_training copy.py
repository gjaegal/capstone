import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import wandb

class GridWorld:
    """Grid world environment with partial observability"""
    def __init__(self, width=24, height=12, n_other_agents=5, view_distance=5):
        self.width = width
        self.height = height
        self.n_other_agents = n_other_agents
        self.view_distance = view_distance
        self.view_width = 5
        self.num_collisions = 0
        self.fixed_positions = False
        
        # Directions: 0=North, 1=East, 2=South, 3=West
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        self.reset()
    
    def reset(self):
        """Reset environment"""


        # Main agent position and direction
        # self.agent_pos = [self.width // 2, 0]
        if self.fixed_positions:
            self.agent_pos = [self.width // 2, self.height - 1]
            self.agent_dir = 0  # Always start facing North
        else:
            self.agent_pos = [np.random.randint(self.width), np.random.randint(self.height)]
            self.agent_dir = np.random.randint(4)
        
        # Other agents positions and directions
        self.other_agents = []
        for _ in range(self.n_other_agents):
            # pos = [self.width//4 + np.random.randint(self.width/2), np.random.randint(self.height)]
            pos = [np.random.randint(self.width), np.random.randint(self.height)]
            direction = np.random.randint(4)
            self.other_agents.append({'pos': pos, 'dir': direction})
        
        # Goal position
        if self.fixed_positions:
            self.goal_pos = [self.width // 2, 0]
        else:
            self.goal_pos = [np.random.randint(self.width), np.random.randint(self.height)]

        # self.goal_pos = [self.width//4 + np.random.randint(self.width/2), self.height//2 + np.random.randint(self.height/2)]
        
        self.steps = 0
        self.max_steps = 200
        self.num_collisions = 0
        
        return self.get_observation()
    
    def get_observation(self):
        """Get partial observation based on agent's view direction"""
        obs = np.zeros((self.view_distance, self.view_width))
        
        dx, dy = self.directions[self.agent_dir]

        
        # Get visible area in front of agent
        for dist in range(0, self.view_distance):
            for w in range(0, self.view_width):
                if dy == 0:
                    view_x = self.agent_pos[0] + dx * (dist + 1)
                    view_y = self.agent_pos[1] + (w - (self.view_width // 2))
                if dx == 0:
                    view_x = self.agent_pos[0] + (w - (self.view_width // 2))
                    view_y = self.agent_pos[1] + dy * (dist+1)
            
                if 0 <= view_x < self.width and 0 <= view_y < self.height:
                    # Check for other agents
                    for other in self.other_agents:
                        if other['pos'] == [view_x, view_y]:
                            obs[dist, w] = 1
                    
                    # Check for goal
                    if self.goal_pos == [view_x, view_y]:
                        obs[dist, w] = 2
                else:
                    obs[dist, w] = -1  # Wall
        
        # Add agent's own position and direction info
        obs_flat = obs.flatten()
        
        # Add relative goal direction info
        rel_goal_x = (self.goal_pos[0] - self.agent_pos[0]) / self.width
        rel_goal_y = (self.goal_pos[1] - self.agent_pos[1]) / self.height
        # Append normalized position and direction
        agent_info = np.array([
            self.agent_pos[0] / self.width,
            self.agent_pos[1] / self.height,
            self.agent_dir / 4.0,
            rel_goal_x,
            rel_goal_y
        ])
        
        return np.concatenate([obs_flat, agent_info])
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        self.steps += 1
        
        # Execute agent action
        reward = -0.01  # Small step penalty
        
        if action == 0:  # Go forward
            dx, dy = self.directions[self.agent_dir]
            new_x = self.agent_pos[0] + dx
            new_y = self.agent_pos[1] + dy
            if 0 <= new_x < self.width and 0 <= new_y < self.height:
                self.agent_pos = [new_x, new_y]
            else:
                reward = -0.5  # Wall collision penalty
                
        elif action == 1:  # Turn 180
            self.agent_dir = (self.agent_dir + 2) % 4
                
        elif action == 2:  # Turn right
            self.agent_dir = (self.agent_dir + 1) % 4
            
        elif action == 3:  # Turn left
            self.agent_dir = (self.agent_dir - 1) % 4
        
        # Move other agents randomly
        for other in self.other_agents:
            other_action = np.random.randint(4)
            if other_action == 0:  # Forward
                dx, dy = self.directions[other['dir']]
                new_x = other['pos'][0] + dx
                new_y = other['pos'][1] + dy
                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    other['pos'] = [new_x, new_y]
            elif other_action == 1:  # Turn 180
                other['dir'] = (other['dir'] + 2) % 4
            elif other_action == 2:  # Turn right
                other['dir'] = (other['dir'] + 1) % 4
            elif other_action == 3:  # Turn left
                other['dir'] = (other['dir'] - 1) % 4
        
        # Check if reached goal
        done = False
        if self.agent_pos == self.goal_pos:
            reward = 20.0
            done = True
        
        # Check collision with other agents
        for other in self.other_agents:
            if self.agent_pos == other['pos']:
                reward = -3.0
                self.num_collisions += 1
        
        # Check max steps
        if self.steps >= self.max_steps:
            done = True
            reward = -10.0  # Penalty for not reaching goal in time
        
        return self.get_observation(), reward, done
    


class DRQN(nn.Module):
    """Deep Recurrent Q-Network"""
    def __init__(self, input_size, hidden_size, num_actions):
        super(DRQN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Feature extraction
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # LSTM layer
        self.lstm = nn.LSTM(128, hidden_size, batch_first=True)
        
        # Q-value output
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, num_actions)
        
    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len, input_size)
        hidden: tuple of (h, c) each (1, batch, hidden_size)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Flatten for feature extraction
        x = x.view(batch_size * seq_len, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Reshape for LSTM
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        
        # Q-values for last timestep
        x = x[:, -1, :]  # Take last timestep
        x = torch.relu(self.fc3(x))
        q_values = self.fc4(x)
        
        return q_values, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        h = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return (h, c)


class ReplayBuffer:
    """Experience replay buffer for sequences"""
    def __init__(self, capacity, seq_len):
        self.buffer = deque(maxlen=capacity)
        self.seq_len = seq_len
    
    def push(self, episode):
        """Store an episode"""
        self.buffer.append(episode)
    
    def sample(self, batch_size):
        """Sample batch of sequences"""
        episodes = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        for episode in episodes:
            states, actions, rewards, next_states, dones = episode
            
            # Sample random starting point
            if len(states) > self.seq_len:
                start_idx = np.random.randint(0, len(states) - self.seq_len + 1)
                end_idx = start_idx + self.seq_len
            else:
                start_idx = 0
                end_idx = len(states)
            
            # Pad if necessary
            seq_states = states[start_idx:end_idx]
            seq_actions = actions[start_idx:end_idx]
            seq_rewards = rewards[start_idx:end_idx]
            seq_next_states = next_states[start_idx:end_idx]
            seq_dones = dones[start_idx:end_idx]
            
            # Pad sequences if needed
            while len(seq_states) < self.seq_len:
                seq_states.append(np.zeros_like(states[0]))
                seq_actions.append(0)
                seq_rewards.append(0)
                seq_next_states.append(np.zeros_like(states[0]))
                seq_dones.append(True)
            
            batch_states.append(seq_states)
            batch_actions.append(seq_actions)
            batch_rewards.append(seq_rewards)
            batch_next_states.append(seq_next_states)
            batch_dones.append(seq_dones)
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
    
    def __len__(self):
        return len(self.buffer)


def train_drqn():
    """Main training loop"""
    # Hyperparameters
    num_episodes = 3000
    batch_size = 32
    seq_len = 16
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995
    learning_rate = 0.0001
    target_update = 10
    buffer_capacity = 1000
    hidden_size = 64
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Environment
    env = GridWorld(width=24, height=12, n_other_agents=3, view_distance=8)
    obs_size = len(env.reset())
    num_actions = 4
    
    # Networks
    policy_net = DRQN(obs_size, hidden_size, num_actions).to(device)
    target_net = DRQN(obs_size, hidden_size, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_capacity, seq_len)
    
    epsilon = epsilon_start
    episode_rewards = []
    
    print("Starting training...")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Episode storage
        episode_states = []
        episode_actions = []
        episode_rewards_list = []
        episode_next_states = []
        episode_dones = []
        episode_collisions = []
        
        hidden = None
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(num_actions) 
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
                    if hidden is None:
                        hidden = policy_net.init_hidden(1, device)
                    q_values, hidden = policy_net(state_tensor, hidden)
                    action = q_values.argmax().item()
            
            next_state, reward, done = env.step(action)
            
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards_list.append(reward)
            episode_next_states.append(next_state)
            episode_dones.append(done)

            
            state = next_state
            episode_reward += reward

        episode_collisions.append(env.num_collisions)
        wandb.log({"episode_collisions": env.num_collisions}, step=episode)
        
        # Store episode
        replay_buffer.push((episode_states, episode_actions, episode_rewards_list, 
                           episode_next_states, episode_dones))
        
        episode_rewards.append(episode_reward)
        wandb.log({"episode_reward": episode_reward, "epsilon": epsilon}, step=episode)
        
        # Training
        if len(replay_buffer) >= batch_size:
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)
            
            # Convert to tensors
            states_tensor = torch.FloatTensor(batch_states).to(device)
            actions_tensor = torch.LongTensor(batch_actions).to(device)
            rewards_tensor = torch.FloatTensor(batch_rewards).to(device)
            next_states_tensor = torch.FloatTensor(batch_next_states).to(device)
            dones_tensor = torch.FloatTensor(batch_dones).to(device)
            
            # Current Q values
            hidden_policy = policy_net.init_hidden(batch_size, device)
            q_values, _ = policy_net(states_tensor, hidden_policy)
            q_values = q_values.gather(1, actions_tensor[:, -1].unsqueeze(1))
            
            # Target Q values
            with torch.no_grad():
                hidden_target = target_net.init_hidden(batch_size, device)
                next_q_values, _ = target_net(next_states_tensor, hidden_target)
                next_q_values = next_q_values.max(1)[0]
                target_q_values = rewards_tensor[:, -1] + gamma * next_q_values * (1 - dones_tensor[:, -1])
            
            # Loss
            loss = nn.MSELoss()(q_values.squeeze(), target_q_values)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()
        
        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Buffer Size: {len(replay_buffer)}")
            wandb.log({"avg_reward_50": avg_reward}, step=episode)
    
    # Save model
    torch.save(policy_net.state_dict(), 'drqn_gridworld.pth')
    print("Training complete! Model saved as 'drqn_gridworld.pth'")
    
    return policy_net, episode_rewards


if __name__ == "__main__":
    wandb.init(project="drqn-gridworld", name="DRQN-Training")
    trained_model, rewards = train_drqn()
    wandb.finish()
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, alpha=0.3)
        plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'))
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('DRQN Training Progress')
        plt.savefig('training_progress.png')
        print("Training plot saved as 'training_progress.png'")
    except ImportError:
        print("Matplotlib not available for plotting")
