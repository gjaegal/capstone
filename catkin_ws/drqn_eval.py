import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Wedge
import time

# Import the environment and model from training script
# Assuming the training script is saved as drqn_training.py
try:
    from drqn_training import GridWorld, DRQN
except ImportError:
    print("Error: Could not import from drqn_training.py")
    print("Make sure the training script is saved as 'drqn_training.py' in the same directory")
    exit(1)


class GridWorldVisualizer:
    """Visualizer for grid world episodes"""
    
    def __init__(self, env, model, device):
        self.env = env
        self.model = model
        self.device = device
        self.model.eval()
        
    def run_episode(self, render_delay=0.1):
        """Run a single episode and collect frames"""
        state = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        frames = []
        hidden = None
        
        while not done:
            # Get current frame
            frame = self.capture_frame()
            frames.append(frame)
            
            # Select action using trained model
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
                if hidden is None:
                    hidden = self.model.init_hidden(1, self.device)
                q_values, hidden = self.model(state_tensor, hidden)
                action = q_values.argmax().item()
            
            # Take action
            next_state, reward, done = self.env.step(action)
            
            total_reward += reward
            steps += 1
            state = next_state
        
        # Capture final frame
        frames.append(self.capture_frame())
        
        return frames, total_reward, steps
    
    def capture_frame(self):
        """Capture current state as a frame dictionary"""
        return {
            'agent_pos': self.env.agent_pos.copy(),
            'agent_dir': self.env.agent_dir,
            'other_agents': [{'pos': a['pos'].copy(), 'dir': a['dir']} 
                            for a in self.env.other_agents],
            'goal_pos': self.env.goal_pos.copy(),
            'steps': self.env.steps
        }
    
    def visualize_static(self, episodes=5, save_path='episodes_visualization.png'):
        """Create static visualization of multiple episodes"""
        fig, axes = plt.subplots(1, episodes, figsize=(4*episodes, 5))
        
        if episodes == 1:
            axes = [axes]
        
        print(f"\nRunning and visualizing {episodes} episodes...")
        
        for ep in range(episodes):
            frames, reward, steps = self.run_episode()
            
            # Visualize final state
            curr_ax = axes[ep]
            self.draw_frame(curr_ax, frames[-1])
            
            curr_ax.set_title(f'Episode {ep+1}\nReward: {reward:.2f} | Steps: {steps}', 
                        fontsize=10, fontweight='bold')
            curr_ax.set_xlim(-0.5, self.env.width - 0.5)
            curr_ax.set_ylim(-0.5, self.env.height - 0.5)
            curr_ax.set_aspect('equal')
            curr_ax.grid(True, alpha=0.3)
            curr_ax.set_xlabel('X')
            curr_ax.set_ylabel('Y')
            curr_ax.invert_yaxis()
            
            print(f"  Episode {ep+1}: Reward={reward:.2f}, Steps={steps}")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nStatic visualization saved to '{save_path}'")
        plt.show()
    
    def draw_frame(self, axis, frame):
        """Draw a single frame on the given axis"""
        # Clear axis
        axis.clear()
        
        # Draw grid
        for i in range(self.env.width + 1):
            axis.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        for i in range(self.env.height + 1):
            axis.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        
        # Draw goal
        goal_x, goal_y = frame['goal_pos']
        star = axis.scatter(goal_x, goal_y, marker='*', s=500, c='gold', 
                         edgecolors='orange', linewidths=2, zorder=3, label='Goal')
        
        # Draw other agents
        for other in frame['other_agents']:
            ox, oy = other['pos']
            other_dir = other['dir']
            
            # Draw circle for other agent
            circle = plt.Circle((ox, oy), 0.3, color='red', alpha=0.6, zorder=2)
            axis.add_patch(circle)
            
            # Draw direction indicator
            dx, dy = [(0, -0.4), (0.4, 0), (0, 0.4), (-0.4, 0)][other_dir]
            axis.arrow(ox, oy, dx, dy, head_width=0.15, head_length=0.1, 
                    fc='darkred', ec='darkred', zorder=2)
        
        # Draw main agent with viewing cone
        agent_x, agent_y = frame['agent_pos']
        agent_dir = frame['agent_dir']
        
        # Draw viewing cone
        cone_angle = 60  # degrees
        direction_angles = [270, 0, 90, 180]  # North, East, South, West in degrees
        base_angle = direction_angles[agent_dir]
        
        wedge = Wedge((agent_x, agent_y), self.env.view_distance, 
                     base_angle - cone_angle/2, base_angle + cone_angle/2,
                     facecolor='blue', alpha=0.2, edgecolor='blue', 
                     linewidth=1, zorder=1, label='View Cone')
        axis.add_patch(wedge)
        
        # Draw main agent
        circle = plt.Circle((agent_x, agent_y), 0.35, color='blue', zorder=4)
        axis.add_patch(circle)
        
        # Draw direction arrow
        dx, dy = [(0, -0.5), (0.5, 0), (0, 0.5), (-0.5, 0)][agent_dir]
        axis.arrow(agent_x, agent_y, dx, dy, head_width=0.2, head_length=0.15, 
                fc='white', ec='white', zorder=5, linewidth=2)
        
        # Add legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=10, label='Main Agent'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Other Agents', alpha=0.6),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                      markersize=15, label='Goal')
        ]
        axis.legend(handles=handles, loc='upper right', fontsize=8)
    
    def create_animation(self, save_path='episode_animation.gif', fps=5):
        """Create animated GIF of a single episode"""
        print("\nCreating animated visualization...")
        
        frames, reward, steps = self.run_episode()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        def update(frame_idx):
            ax.clear()
            frame = frames[frame_idx]
            self.draw_frame(ax, frame)
            
            ax.set_title(f'Step: {frame["steps"]}/{steps} | Total Reward: {reward:.2f}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlim(-0.5, self.env.width - 0.5)
            ax.set_ylim(-0.5, self.env.height - 0.5)
            ax.set_aspect('equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.invert_yaxis()
        
        anim = animation.FuncAnimation(fig, update, frames=len(frames), 
                                      interval=1000//fps, repeat=True)
        
        # Save animation
        try:
            anim.save(save_path, writer='pillow', fps=fps)
            print(f"Animation saved to '{save_path}'")
        except Exception as e:
            print(f"Could not save animation: {e}")
            print("Showing animation instead...")
            plt.show()
        
        plt.close()


def main():
    """Main function to load model and visualize episodes"""
    
    # Check if model file exists
    import os
    if not os.path.exists('drqn_gridworld.pth'):
        print("Error: Model file 'drqn_gridworld.pth' not found!")
        print("Please train the model first using the training script.")
        return
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = GridWorld(width=24, height=12, n_other_agents=3, view_distance=8)
    env.fixed_positions = True  # Use fixed positions
    obs_size = len(env.reset())
    num_actions = 4
    hidden_size = 64
    
    # Load trained model
    print("Loading trained model...")
    model = DRQN(obs_size, hidden_size, num_actions).to(device)
    model.load_state_dict(torch.load('drqn_gridworld.pth', map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    # Create visualizer
    visualizer = GridWorldVisualizer(env, model, device)
    
    # Create static visualization of 5 episodes
    print("\n" + "="*60)
    print("Creating static visualization of 5 episodes...")
    print("="*60)
    visualizer.visualize_static(episodes=5, save_path='episodes_visualization.png')
    
    # Create animation of 1 episode
    print("\n" + "="*60)
    print("Creating animation of 1 episode...")
    print("="*60)
    response = input("Create animation? (y/n): ").strip().lower()
    
    if response == 'y':
        visualizer.create_animation(save_path='episode_animation.gif', fps=5)
    else:
        print("Skipping animation creation.")
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    print("\nOutput files:")
    print("  - episodes_visualization.png: Static view of 5 episode endings")
    if response == 'y':
        print("  - episode_animation.gif: Animated playback of 1 episode")


if __name__ == "__main__":
    main()
