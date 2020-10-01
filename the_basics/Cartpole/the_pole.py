import gym
import numpy as np
from gym import wrappers

env = gym.make('CartPole-v0')
done = False

best_weights = np.zeros(4)
best_score = 0
episode_score = []

observation = env.reset()
 
for i in range(100):
    random_weights = np.random.uniform(-1.0,1.0, 4)
    score = []
    for x in range(100):
        observation = env.reset()
        done = False
        cnt = 0

        while not done:
            #env.render()
            cnt += 1
            action = 1 if np.dot(observation, random_weights) > 0 else 0 

            observation, reward, done, _ = env.step(action)

            if done:
                break
        score.append(cnt)
    average_score = float(sum(score)/ len(score))

    if average_score > best_score:
        best_score = average_score
        best_weights = random_weights
    episode_score.append(average_score)
    if i % 10 == 0:
        print('melhor resultado foi', best_score)
    
done = False
cnt = 0
env = wrappers.Monitor(env, 'ArquivosVideo2',force = True)
observation = env.reset()

while not done:
    cnt += 1
    action = 1 if np.dot(observation, best_weights) > 0 else 0
    observation, reward, done, _ = env.step(action)
    if done:
        break

print(f'O jogo durou {cnt} jogadas')