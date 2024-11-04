import gymnasium as gym
import os
import numpy as np
import torch
from torch import nn

from modules import PolicyNet, ValueNet
from simple_network import SimpleNet
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from create_gif import create_failure_gif, create_success_gif

try:
    import wandb
except ImportError:
    wandb = None

class TrainDaggerBC:

    def __init__(self, env, model, expert_model, optimizer, states, actions, device="cpu", mode="DAgger"):
        """
        Initializes the TrainDAgger class. Creates necessary data structures.

        Args:
            env: an OpenAI Gym environment.
            model: the model to be trained.
            expert_model: the expert model that provides the expert actions.
            device: the device to be used for training.
            mode: the mode to be used for training. Either "DAgger" or "BC".

        """
        self.env = env
        self.model = model
        self.expert_model = expert_model
        self.optimizer = optimizer
        self.device = device
        model.set_device(self.device)
        self.expert_model = self.expert_model.to(self.device)

        self.mode = mode

        if self.mode == "BC":
            self.states = []
            self.actions = []
            self.timesteps = []
            for trajectory in range(states.shape[0]):
                trajectory_mask = states[trajectory].sum(axis=1) != 0
                self.states.append(states[trajectory][trajectory_mask])
                self.actions.append(actions[trajectory][trajectory_mask])
                self.timesteps.append(np.arange(0, len(trajectory_mask)))
            self.states = np.concatenate(self.states, axis=0)
            self.actions = np.concatenate(self.actions, axis=0)
            self.timesteps = np.concatenate(self.timesteps, axis=0)

            self.clip_sample_range = 1
            self.actions = np.clip(self.actions, -self.clip_sample_range, self.clip_sample_range)

        else:
            self.states = None
            self.actions = None
            self.timesteps = None

    def generate_trajectory(self, env, policy):
        """Collects one rollout from the policy in an environment. The environment
        should implement the OpenAI Gym interface. A rollout ends when done=True. The
        number of states and actions should be the same, so you should not include
        the final state when done=True.

        Args:
            env: an OpenAI Gym environment.
            policy: The output of a deep neural network
            Returns:
            states: a list of states visited by the agent.
            actions: a list of actions taken by the agent. Note that these actions should never actually be trained on...
            timesteps: a list of integers, where timesteps[i] is the timestep at which states[i] was visited.
        """

        states, old_actions, timesteps, rewards = [], [], [], []

        done, trunc = False, False
        cur_state, _ = env.reset()  
        t = 0
        while (not done) and (not trunc):
            with torch.no_grad():
                p = policy(torch.from_numpy(cur_state).to(self.device).float().unsqueeze(0), torch.tensor(t).to(self.device).long().unsqueeze(0))
            a = p.cpu().numpy()[0]
            next_state, reward, done, trunc, _ = env.step(a)

            states.append(cur_state)
            old_actions.append(a)
            timesteps.append(t)
            rewards.append(reward)

            t += 1

            cur_state = next_state

        return states, old_actions, timesteps, rewards

    def call_expert_policy(self, state):
        """
        Calls the expert policy to get an action.

        Args:
            state: the current state of the environment.
        """
        # takes in a np array state and returns an np array action
        with torch.no_grad():
            state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=self.device)
            action = self.expert_model.choose_action(state_tensor, deterministic=True).cpu().numpy()
            action = np.clip(action, -1, 1)[0]
        return action

    def update_training_data(self, num_trajectories_per_batch_collection=20):
        """
        Updates the training data by collecting trajectories from the current policy and the expert policy.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.

        NOTE: you will need to call self.generate_trajectory and self.call_expert_policy in this function.
        NOTE: you should update self.states, self.actions, and self.timesteps in this function.
        """
        # BEGIN STUDENT SOLUTION
        for _ in range(num_trajectories_per_batch_collection):
            states, actions, timesteps, rewards = self.generate_trajectory(self.env, self.model)
            expert_actions = [self.call_expert_policy(state) for state in states]
            if self.states is None:
                self.states = states
                self.actions = expert_actions
                self.timesteps = timesteps
            else:
                self.states = np.concatenate([self.states, states], axis=0)
                self.actions = np.concatenate([self.actions, expert_actions], axis=0)
                self.timesteps = np.concatenate([self.timesteps, timesteps], axis=0)
        # END STUDENT SOLUTION

        return rewards

    def generate_trajectories(self, num_trajectories_per_batch_collection=20):
        """
        Runs inference for a certain number of trajectories. Use for behavior cloning.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.
        
        NOTE: you will need to call self.generate_trajectory in this function.
        """
        # BEGIN STUDENT SOLUTION
        
        total_rewards = []
        for _ in range(num_trajectories_per_batch_collection):
            _, _, _, rewards = self.generate_trajectory(self.env, self.model)
            total_rewards.append(sum(rewards))
        # END STUDENT SOLUTION

        return total_rewards

    def train(
        self, 
        num_batch_collection_steps=20, 
        num_BC_training_steps=20000,
        num_training_steps_per_batch_collection=1000, 
        num_trajectories_per_batch_collection=20, 
        batch_size=64, 
        print_every=1000, 
        save_every=10000, 
        wandb_logging=False
    ):
        """
        Train the model using BC or DAgger

        Args:
            num_batch_collection_steps: the number of times to collect a batch of trajectories from the current policy.
            num_BC_training_steps: the number of iterations to train the model using BC.
            num_training_steps_per_batch_collection: the number of times to train the model per batch collection.
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy per batch.
            batch_size: the batch size to use for training.
            print_every: how often to print the loss during training.
            save_every: how often to save the model during training.
            wandb_logging: whether to log the training to wandb.

        NOTE: for BC, you will need to call the self.training_step function and self.generate_trajectories function.
        NOTE: for DAgger, you will need to call the self.training_step and self.update_training_data function.
        """

        # BEGIN STUDENT SOLUTION
        losses = []
        rewards = []
        evaluate_reward_every = print_every
        if self.mode == "BC":
            for step in range(1, num_BC_training_steps + 1):
                loss = self.training_step(batch_size)
                losses.append(loss)
                    
                if step % print_every == 0:
                    print(f"Step {step}: Loss = {loss:.4f}")
                
                if step % evaluate_reward_every == 0:
                    print('start evaluating')
                    # Evaluate the policy by generating trajectories
                    traj_rewards = self.generate_trajectories(num_trajectories_per_batch_collection=num_trajectories_per_batch_collection)
                    avg_reward = np.mean(traj_rewards)
                    median_reward = np.median(traj_rewards)
                    max_reward = np.max(traj_rewards)
                    rewards.append({'step': step, 'loss': loss, 'avg_reward': avg_reward, 'median_reward': median_reward, 'max_reward': max_reward})

                    print(f"Step {step}: Loss = {loss:.4f}, Avg Reward = {avg_reward:.4f}, Median Reward = {median_reward:.4f}, Max Reward = {max_reward:.4f}")
                else:
                    rewards.append({'step': step, 'loss': loss})
                if step % save_every == 0:
                    torch.save(self.model.state_dict(), f"data/models/bc.pt")

        elif self.mode == "DAgger":
            step = 0
            for step_out in range(1, num_batch_collection_steps + 1):
                self.update_training_data(num_trajectories_per_batch_collection=num_trajectories_per_batch_collection)
                for _ in range(num_training_steps_per_batch_collection):
                    loss = self.training_step(batch_size)
                    step+=1
                    losses.append(loss)
            
                    if step % print_every == 0:
                        print(f"Step {step}: Loss = {loss:.4f}")
                    if step % evaluate_reward_every == 0:
                        print('start evaluating')
                        # Evaluate the policy by generating trajectories
                        traj_rewards = self.generate_trajectories(num_trajectories_per_batch_collection=num_trajectories_per_batch_collection)
                        avg_reward = np.mean(traj_rewards)
                        median_reward = np.median(traj_rewards)
                        max_reward = np.max(traj_rewards)
                        rewards.append({'step': step, 'loss': loss,'avg_reward': avg_reward, 'median_reward': median_reward, 'max_reward': max_reward})

                        print(f"Step {step}: Loss = {loss:.4f}, Avg Reward = {avg_reward:.4f}, Median Reward = {median_reward:.4f}, Max Reward = {max_reward:.4f}")
                    else:
                        rewards.append({'step': step, 'loss': loss})

                if step % save_every == 0:
                    torch.save(self.model.state_dict(), f"data/models/dagger.pt")
            # DAgger code would go here (not required for this task)
            
        # END STUDENT SOLUTION

        return losses, rewards

    def training_step(self, batch_size):
        """
        Simple training step implementation

        Args:
            batch_size: the batch size to use for training.
        """
        states, actions, timesteps = self.get_training_batch(batch_size=batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        timesteps = timesteps.to(self.device)

        loss_fn = nn.MSELoss()
        self.optimizer.zero_grad()
        predicted_actions = self.model(states, timesteps)
        loss = loss_fn(predicted_actions, actions)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def get_training_batch(self, batch_size=64):
        """
        get a training batch

        Args:
            batch_size: the batch size to use for training.
        """
        # get random states, actions, and timesteps
        indices = np.random.choice(len(self.states), size=batch_size, replace=False)
        states = torch.tensor(self.states[indices], device=self.device, dtype=torch.float32)
        actions = torch.tensor(self.actions[indices], device=self.device, dtype=torch.float32)
        timesteps = torch.tensor(self.timesteps[indices], device=self.device)
            
        
        return states, actions, timesteps

def run_training(mode="BC"):
    """
    Simple Run Training Function
    """

    env = gym.make('BipedalWalker-v3') # , render_mode="rgb_array"

    model_name = "super_expert_PPO_model"
    expert_model = PolicyNet(24, 4)
    model_weights = torch.load(f"data/models/{model_name}.pt", map_location=torch.device('mps') )
    expert_model.load_state_dict(model_weights["PolicyNet"])

    states_path = "data/states_BC.pkl"
    actions_path = "data/actions_BC.pkl"

    with open(states_path, "rb") as f:
        states = pickle.load(f)
    with open(actions_path, "rb") as f:
        actions = pickle.load(f)

    # BEGIN STUDENT SOLUTION
    model = SimpleNet(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_layer_dimension=128,
        max_episode_length=1600, 
        device = "mps"
    )

    # Use the AdamW optimizer with the specified parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

    # Create the trainer object
    trainer = TrainDaggerBC(
        env=env,
        model=model,
        expert_model=expert_model,
        optimizer=optimizer,
        states=states,
        actions=actions,
        device="mps",
        mode=mode
    )

    # Train the model
    if mode == "BC":
        losses, rewards = trainer.train(
            num_BC_training_steps=20000,
            batch_size=64,
            print_every=1000,
            save_every=1000
        )
    elif mode == "DAgger":
        losses, rewards = trainer.train(
            num_batch_collection_steps=20,
            num_training_steps_per_batch_collection=1000,
            num_trajectories_per_batch_collection=20,
            batch_size=128,
            print_every=1000,
            save_every=1000
        )
    

    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title(f'Training Loss with {mode}')
    plt.savefig(f'training_loss_plot_{mode}.pdf')
    plt.close()

    # Write down the final loss value
    final_loss = losses[-1]
    print(f"Final Loss Value: {final_loss}")

    # Create a graph of the average, median, and max rewards over time
    losses = [r['loss'] for r in rewards]
    steps = [r['step'] for r in rewards if 'avg_reward' in r]
    avg_rewards = [r['avg_reward'] for r in rewards if 'avg_reward' in r]
    median_rewards = [r['median_reward'] for r in rewards if 'avg_reward' in r]
    max_rewards = [r['max_reward'] for r in rewards if 'avg_reward' in r]


    import pandas as pd
    # import code; code.interact(local=dict(globals(), **locals()))
    data = pd.DataFrame(rewards)
    # data = pd.DataFrame({'step': steps, 'loss': losses, 'avg_reward': avg_rewards, 'median_reward': median_rewards, 'max_reward': max_rewards})
    data.to_csv(f'policy_performance_data_{mode}.csv', index=False)

    plt.figure()
    plt.plot(steps, avg_rewards, label='Average Reward')
    plt.plot(steps, median_rewards, label='Median Reward')
    plt.plot(steps, max_rewards, label='Max Reward')
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')
    plt.title('Policy Performance over Time')
    plt.legend()
    plt.savefig(f'policy_performance_plot_{mode}.pdf')
    plt.close()

    env = gym.make('BipedalWalker-v3', render_mode="rgb_array")
    if mode == "BC":
        create_failure_gif(env, model, max_episode_length=1600, device="mps")
    elif mode == "DAgger":
        create_success_gif(env, model, max_episode_length=1600, device="mps")

    # END STUDENT SOLUTION

if __name__ == "__main__":
    # run_training("BC")
    # print("Finished training BC")
    print("Starting training DAgger")
    run_training("DAgger")
    print("Finished training DAgger")