import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

maxstates = 10 ** 4
gamma = 0.9
alpha = 0.01

def max_dict(d):
    max_v = float('-inf')
    for key, val in d.items():
        if val > max_v:
            max_v = val
            max_key = key
    return max_key, max_v

def create_bins():
    bins = np.zeros((4,10))
    bins[0] = np.linspace(-4.8, 4.8, 10)
    bins[1] = np.linspace(-5, 5, 10)
    bins[2] = np.linspace(-.418, .418, 10)
    bins[3] = np.linspace(-5, 5, 10)

    return bins

def assing_bins(observation, bins):
    state = np.zeros(4)
    for i in range(4):
        state[i] = np.digitize(observation[i], bins[i])
    return state

def get_state_as_string(state):
    string_state = ''.join(str(int(e)) for e in state)

def get_all_states_as_string():
    states = []
    for i in range(maxstates):
        states.append(str(i).zfill(4))
    return states

def initialize_Q():
    Q = {}

    all_states = get_all_states_as_string()

    for state in all_states:
        Q[state] = {}
        for action in range([env.action_space.n]):
            Q[state][action] = 0
    return Q

def play_one_game(bins, Q, eps=0.5):
    observation = env.reset()
    done = False
    cnt = 0 
    state= get_state_as_string(assing_bins(observation,bins))
    total_reward = 0

    while not done:
        cnt += 1
        if np.random.uniform() < eps: 
            act = env.action_space.sample()
        else:
            act = max_dict(Q[state])[0]
           
        observation,reward, done, _ = env.step(act)
        
        total_reward = reward

        if done and cnt < 200:
            reward = 300

        state_new = get_all_states_as_string(assing_bins(observation, bins))

        a1, max_q_sla1 = max_dict(Q[state_new])
        Q[state][act] += alpha * (reward + gamma * max_q_sla1 - Q[state][act])
        state, act = state_new, a1
    
    return total_reward, cnt
    
def play_many_games(bins, N=10000):
	Q = initialize_Q()

	length = []
	reward = []
	for n in range(N):
		#eps=0.5/(1+n*10e-3)
		eps = 1.0 / np.sqrt(n+1)

		episode_reward, episode_length = play_one_game(bins, Q, eps)
		
		if n % 100 == 0:
			print(n, '%.4f' % eps, episode_reward)
		length.append(episode_length)
		reward.append(episode_reward)

	return length, reward

def plot_running_avg(totalrewards):
	N = len(totalrewards)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
	plt.plot(running_avg)
	plt.title("Running Average")
	plt.show()

if __name__ == '__main__':
	bins = create_bins()
	episode_lengths, episode_rewards = play_many_games(bins)
