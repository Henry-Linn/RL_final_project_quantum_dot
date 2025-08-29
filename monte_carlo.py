import numpy as np
import main
import random
import matplotlib.pyplot as plt

states = main.gen_possStates()
actions = main.get_possActions(3, 6)

state_to_index = {}
index_to_state = {}

action_to_index = {}
index_to_action = {}

q1 = [0,1,2,3]
q2 = [0,1,4,5]
q3 = [0,2,4,6]
num_unique = [q1, q2, q3]
binary_maps = main.get_binary(num_unique)

for i, state in enumerate(states):
    state_to_index[state] = i
    index_to_state[i] = state

for i, action in enumerate(actions):
    action_to_index[action] = i
    index_to_action[i] = action  

def MC_epPolicy(epsilon):
    mapping_actions = dict(zip([i for i in range(len(actions))], actions))
    mapping_states = dict(zip(states,[i for i in range(len(states))]))
    def initialize():
        q = np.zeros((len(states), len(actions)))
        state_distribution = np.random.rand(len(states), len(actions))
        sum_cols = np.sum(state_distribution, axis=1, keepdims=True)
        state_distribution = (state_distribution/sum_cols)
        pi = dict(zip(states, state_distribution))

        empty_lists = [[[] for i in range(len(actions))] for j in range(len(states))]
        returns = dict(zip(states, empty_lists))
        return q, pi, returns
    

    def get_OneEpisode(pi):
        s = (0,0,0,0,0,0)
        orr_states = set()
        states = []
        rewards = []
        # checking if it is a terminal state
        while 0 in s:
            weights = pi[s]
            action = random.choices(np.arange(len(actions)), weights = weights)[0]
            s_prime = main.take_step(s, mapping_actions[action])
            orr_states.add(s)
            if (s_prime) not in orr_states:  
                reward = main.get_reward(s, s_prime)
                states.append((s, action))
                rewards.append(reward)
                s = s_prime
        return states, rewards
    
    def cal_v(pi, q):
        v = np.zeros(len(states))
        for state in states:
            v[mapping_states[state]] = np.sum(pi[state]*q[mapping_states[state]])
        return v

    q, pi, returns = initialize()
    theta = 0.0001
    delta = 999999999
    num_eps = 1000
    iteration = 0
    R = []
    while delta > theta or (delta == 0):
        iteration += 1
        ep_states, ep_rewards = get_OneEpisode(pi)
        prev = cal_v(pi, q)
        for i, one_state in enumerate(ep_states):
            state = one_state[0]
            action = one_state[1]
            returns[state][action].append(np.sum(ep_rewards[i:]))
            q[mapping_states[state]][action] = np.mean(returns[state][action])
            a_star = np.argmax(q[mapping_states[state]])
            # define the distribution
            state_distribution = np.ones(len(actions))*epsilon/(len(actions))
            state_distribution[a_star] = 1 - epsilon + epsilon/len(actions)
            pi[state] = state_distribution.copy()
        epsilon = 0.9*epsilon
        R.append(np.sum(ep_rewards))
        v = cal_v(pi, q)
        delta = np.max(np.abs(v - prev))
    plt.figure()
    x = np.arange(len(R))
    plt.xlabel("episode")
    plt.ylabel("J")
    plt.plot(x, R)
    plt.show()
    return q, iteration

def test_policy(q_table):
    state = (0,0,0,0,0,0)
    for _ in range(10):
        main.print_state(state)
        action = index_to_action[np.argmax(q_table[state_to_index[state], :])]
        print(action)
        new_state = main.take_step(state, action)
        reward = main.get_reward(state, new_state)
        print("Reward:", reward)
        state = new_state
    print("number of shifts = ", main.get_shift(main.construct_matrixState(state, binary_maps)))

epsilon = 0.9
q_star, iterations = MC_epPolicy(epsilon)
test_policy(q_star)