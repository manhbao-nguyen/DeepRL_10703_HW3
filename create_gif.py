import numpy as np
import torch
from PIL import Image  # Import PIL for image handling

def create_failure_gif(env, policy, max_episode_length=1600, device="mps"):
    while True:
        frames = []
        total_reward = 0
        state, _ = env.reset()
        t = 0
        done = False
        trunc = False

        while not done and not trunc and t < max_episode_length:
            frame = env.render()
            frames.append(frame)

            state_tensor = torch.from_numpy(state).to(device).float().unsqueeze(0)
            timestep_tensor = torch.tensor([t], device=device)
            with torch.no_grad():
                action = policy(state_tensor, timestep_tensor).cpu().numpy()[0]
            next_state, reward, done, trunc, _ = env.step(action)
            total_reward += reward
            state = next_state
            t += 1

        if total_reward < 0:
            print(f"Failure run found with total reward: {total_reward}")
            images = [Image.fromarray(frame) for frame in frames]
            images[0].save(
                'failure_run.gif',
                save_all=True,
                append_images=images[1:],
                duration=50,  # Duration between frames in milliseconds
                loop=0
            )
            print("GIF saved as 'failure_run.gif'")
            break
        else:
            print(f"Run with total reward {total_reward} is not a failure. Retrying...")

def create_success_gif(env, policy, max_episode_length=1600, device="mps"):
    while True:
        frames = []
        total_reward = 0
        state, _ = env.reset()
        t = 0
        done = False
        trunc = False

        while not done and not trunc and t < max_episode_length:
            frame = env.render()
            frames.append(frame)

            state_tensor = torch.from_numpy(state).to(device).float().unsqueeze(0)
            timestep_tensor = torch.tensor([t], device=device)
            with torch.no_grad():
                action = policy(state_tensor, timestep_tensor).cpu().numpy()[0]
            next_state, reward, done, trunc, _ = env.step(action)
            total_reward += reward
            state = next_state
            t += 1

        if total_reward > 260:
            print(f"Success run found with total reward: {total_reward}")
            images = [Image.fromarray(frame) for frame in frames]
            images[0].save(
                'success_run.gif',
                save_all=True,
                append_images=images[1:],
                duration=50, 
                loop=0
            )
            print("GIF saved as 'success_run.gif'")
            break
        else:
            print(f"Run with total reward {total_reward} is not a success. Retrying...")


def create_success_gif_diffusion(
    trainer,
    env,
    num_actions_to_eval_in_a_row=3,
    num_previous_states=5,
    num_previous_actions=4,
    reward_threshold=240,
    max_attempts=10,
    gif_filename='successful_run.gif'
):

    trainer.model.eval()  

    attempt = 0
    success = False

    while attempt < max_attempts and not success:
        attempt += 1
        print(f"Attempt {attempt} to generate a successful run...")

        frames = []
        rewards = []
        s, a, t_list, done, truncated = [], [], [], False, False

        state, _ = env.reset()
        s.append(state)
        t_list.append(0)
        rewards = []
        frames.append(env.render())

        max_sequence_length = trainer.max_trajectory_length

        while not done and not truncated:
            previous_states = s[-num_previous_states:]
            previous_actions = a[-num_previous_actions:]
            episode_timesteps = t_list[-num_previous_states:]

            if len(previous_states) < num_previous_states:
                num_padding = num_previous_states - len(previous_states)
                previous_states_padding_mask = torch.cat([
                    torch.zeros(len(previous_states), dtype=torch.bool),
                    torch.ones(num_padding, dtype=torch.bool)
                ], dim=0)
                previous_states = previous_states + [np.zeros_like(state) for _ in range(num_padding)]
                episode_timesteps = episode_timesteps + [0 for _ in range(num_padding)]
            else:
                previous_states_padding_mask = torch.zeros(num_previous_states, dtype=torch.bool)

            if len(previous_actions) < num_previous_actions:
                num_padding = num_previous_actions - len(previous_actions)
                previous_actions_padding_mask = torch.cat([
                    torch.zeros(len(previous_actions), dtype=torch.bool),
                    torch.ones(num_padding, dtype=torch.bool)
                ], dim=0)
                previous_actions = previous_actions + [np.zeros(trainer.action_dimension) for _ in range(num_padding)]
            else:
                previous_actions_padding_mask = torch.zeros(num_previous_actions, dtype=torch.bool)

            previous_states = np.array(previous_states)
            previous_states = (previous_states - trainer.states_mean) / trainer.states_std
            previous_states = torch.from_numpy(previous_states).float().unsqueeze(0).to(trainer.device)

            previous_actions = np.array(previous_actions)
            if len(previous_actions) > 0:
                previous_actions = (previous_actions - trainer.actions_mean) / trainer.actions_std
            else:
                previous_actions = np.zeros((num_previous_actions, trainer.action_dimension))
            previous_actions = torch.from_numpy(previous_actions).float().unsqueeze(0).to(trainer.device)

            episode_timesteps = torch.tensor(episode_timesteps).long().unsqueeze(0).to(trainer.device)

            previous_states_padding_mask = previous_states_padding_mask.unsqueeze(0).to(trainer.device)
            previous_actions_padding_mask = previous_actions_padding_mask.unsqueeze(0).to(trainer.device)
            actions_padding_mask = torch.zeros(num_actions_to_eval_in_a_row, dtype=torch.bool).unsqueeze(0).to(trainer.device)

            with torch.no_grad():
                predicted_actions = trainer.diffusion_sample(
                    previous_states=previous_states,
                    previous_actions=previous_actions,
                    episode_timesteps=episode_timesteps,
                    previous_states_padding_mask=previous_states_padding_mask,
                    previous_actions_padding_mask=previous_actions_padding_mask,
                    actions_padding_mask=actions_padding_mask,
                    max_action_len=num_actions_to_eval_in_a_row
                )

            predicted_actions = predicted_actions.squeeze(0).cpu().numpy()
            denormalized_actions = predicted_actions * trainer.actions_std + trainer.actions_mean
            denormalized_actions = np.clip(denormalized_actions, -trainer.clip_sample_range, trainer.clip_sample_range)

            for i in range(num_actions_to_eval_in_a_row):
                action = denormalized_actions[i]
                next_state, reward, done, truncated, _ = env.step(action)
                s.append(next_state)
                a.append(action)
                t_list.append(t_list[-1] + 1)
                rewards.append(reward)
                frames.append(env.render())

                if done or truncated:
                    break

            s = s[-max_sequence_length:]
            a = a[-max_sequence_length:]
            t_list = t_list[-max_sequence_length:]

        total_reward = sum(rewards)
        if total_reward >= reward_threshold:
            success = True
            print(f"Successful run found with total reward: {total_reward:.2f}")
            images = [Image.fromarray(frame) for frame in frames]
            images[0].save(
                gif_filename,
                save_all=True,
                append_images=images[1:],
                duration=50,  
                loop=0
            )
            print(f"GIF saved as '{gif_filename}'")
        else:
            print(f"Run did not meet the reward threshold ({total_reward:.2f} < {reward_threshold})")

    if not success:
        print(f"Could not find a successful run in {max_attempts} attempts.")
