import gym
import numpy as np
import matplotlib.pyplot as plt

pos_space = np.linspace(-1.2, 0.6, 20)
vel_space = np.linspace(-0.07, 0.07, 20)

def get_state(observation):
    pos, vel = observation
    pos_bin = int(np.digitize(pos, pos_space))
    vel_bin = int(np.digitize(vel, vel_space))

    return (pos_bin, vel_bin)

def max_action(Q, state, actions=[0, 1, 2]):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)

    return action

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    n_games = 50000
    alfa = 0.1
    gama = 0.99
    epsilon = 1.0

    action_space = [0, 1, 2]

    states = []
    for pos in range(21):
        for vel in range(21):
            states.append((pos,vel))

    Q = {}
    for state in states:
        for action in action_space:
            Q[state, action] = 0

    score = 0
    total_rewards = np.zeros(n_games)
    for i in range(n_games):
        done = False
        obs = env.reset()
        state = get_state(obs)

        if i % 1000 == 0 and i > 0:
            print(f'Episode {i}, score {score}, epsilon {epsilon:.2f}')
        score = 0
        
        while not done:
            action = np.random.choice([0,1,2]) if np.random.random() < epsilon else max_action(Q, state)

            new_obs, reward, done, info = env.step(action)
            new_state = get_state(new_obs)            
            new_action = max_action(Q, new_state)

            score += reward

            Q[state, action] = Q[state, action] + alfa*(reward + gama* Q[new_state, new_action] - Q[state,action])

            state = new_state
        total_rewards[i] = score
        epsilon = epsilon - 2/n_games if epsilon > 0.001 else 0.001
    
    mean_rewards = np.zeros(n_games)
    for t in range(n_games):
        mean_rewards[t] = np.mean(total_rewards[max(0,t-50):(t+1)])
    plt.plot(mean_rewards)
    plt.sabefig('mountaincar.jpg')
