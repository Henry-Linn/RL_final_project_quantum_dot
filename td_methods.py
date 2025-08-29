import main
import random
import numpy as np
import matplotlib.pyplot as plt

states = main.gen_possStates()
actions = main.get_possActions(3,6)

state_to_index = {}
index_to_state = {}

action_to_index = {}
index_to_action = {}

for i, state in enumerate(states):
    state_to_index[state] = i
    index_to_state[i] = state

for i, action in enumerate(actions):
    action_to_index[action] = i
    index_to_action[i] = action   

def d0():
    return (0,0,0,0,0,0)

def policy(s, q):

    return np.argmax(q[state_to_index[s]])

# alpha = 0.1, epsilon 0.9/0.99
def sarsa(alpha = 0.3, epsilon = 0.99, gamma=1):
    q_table = np.zeros((len(states), len(actions)))

    def pi(s, e):
        num = random.random()
        if e < num:
            a = index_to_action[np.argmax(q_table[state_to_index[s], :])]
        else:
            a = random.choice(actions)
        return a

    def sim_oneEp(q):
        s0 = (0,0,0,0,0,0)
        s = s0
        reward = 0
        while np.count_nonzero(s) < 6:
            temp_actions = q_table[state_to_index[s]]
            actions = [i[0] for i in sorted(enumerate(temp_actions), key=lambda x:x[1], reverse=True)]
            for action in actions:
                a = index_to_action[action]
                s_prime = tuple(np.array(s)+np.array(a))
                if main.p(s, a, s_prime):
                    reward += main.get_reward(s, s_prime)
                    s = s_prime
                    break
        return reward
    
    numEpisodes = 220000
    rewards = []
    for i in range(numEpisodes):
        s = d0()
        a = pi(s, epsilon)
        while np.count_nonzero(s) < 6:
            s_prime = main.take_step(s, a)
            r = main.get_reward(s, s_prime)
            a_prime = pi(s_prime, epsilon)
            q_table[state_to_index[s], action_to_index[a]] += alpha*(r+gamma*q_table[state_to_index[s_prime], action_to_index[a_prime]] - q_table[state_to_index[s], action_to_index[a]])
            s = s_prime
            a = a_prime

        rewards.append(sim_oneEp(q_table))
        epsilon *= 0.99999
    x = np.arange(len(rewards))
    plt.figure()
    plt.xlabel("iteration")
    plt.ylabel("J")
    plt.plot(x, rewards)
    plt.show()
    return q_table

def test_policy(q_table):
    state = (0,0,0,0,0,0)
    
    for _ in range(10):
        main.print_state(state)
        action = index_to_action[np.argmax(q_table[state_to_index[state], :])]
        new_state = main.take_step(state, action)
        reward = main.get_reward(state, new_state)
        print("Reward:", reward)
        state = new_state

q_star = sarsa()
test_policy(q_star)