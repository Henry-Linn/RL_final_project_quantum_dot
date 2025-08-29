import main
import numpy as np
import random
import matplotlib.pyplot as plt

states = main.gen_possStates()

def policy(state, v_star):
    actions = main.get_possActions(3,6)
    poss_reward_actions = {}
    for action in actions:
        sum = 0
        s_prime = tuple(np.array(state)+ np.array(action))
        if s_prime in state_to_index:
            #print("inside = ",main.p(state, action, s_prime))
            poss_reward_actions[action] = main.p(state, action, s_prime) * (main.get_reward(state, s_prime) + gamma * v_star[state_to_index[s_prime]])
        else:
            poss_reward_actions[action] = 0
    max_val = max(poss_reward_actions.values())
    poss_maxStates = [k for k,v in poss_reward_actions.items() if v == max_val]
    return random.choice(poss_maxStates)

# Warning: Not tested
def value_iteration(values, gamma, state_to_index):
    q1 = [0,1,2,3]
    q2 = [0,1,4,5]
    q3 = [0,2,4,6]
    num_unique = [q1, q2, q3]

    states = main.gen_possStates()
    actions = main.get_possActions(3,6)

    iteration = 0 
    theta = 0.0001

    def sim_oneEp(v):
        s0 = (0,0,0,0,0,0)
        s = s0
        reward = 0
        for i in range(7):
            a = policy(s, v)
            s_prime = tuple(np.array(s)+np.array(a))
            if main.p(s, a, s_prime):
                reward += main.get_reward(s, s_prime)
                s = s_prime
        return reward

    # equation after the maxA part: sum_s'...
    def equation(state, action):
        sum = 0
        s_prime = tuple(np.array(state)+ np.array(action))
        try:
            sum += main.p(state, action, s_prime) * (main.get_reward(state, s_prime) + gamma * values[state_to_index[s_prime]])
        except KeyError:
            return 0
        return sum
    
    flag = True
    rewards = []
    while flag:
        delta = 0
        for i, state in enumerate(states):
            #print(i)
            v = values[i]
            action_values = np.zeros(len(actions))
            for j, action in enumerate(actions):
                action_values[j] = equation(state, action)

            values[i] = max(action_values)
            delta = max(delta, abs(v-values[i]))
        rewards.append(sim_oneEp(values))
        if delta < theta:
            flag = False
        iteration += 1

    x = np.arange(len(rewards))
    plt.figure()
    plt.xlabel("episode")
    plt.ylabel("J")
    plt.plot(x, rewards)
    plt.show()
    return values

values = np.zeros(len(states))
actions = main.get_possActions(3,6)
state_to_index = {}
for i, v in enumerate(states):
    state_to_index[v] = i

gamma = 1
v_star = value_iteration(values, gamma, state_to_index)

# game loop
def game_loop():
    s0 = (0,0,0,0,0,0)
    s = s0
    for i in range(7):
        main.print_state(s)
        a = policy(s, v_star)
        print("action is ", a)

        s_prime = tuple(np.array(s)+np.array(a))
        s = s_prime

game_loop()