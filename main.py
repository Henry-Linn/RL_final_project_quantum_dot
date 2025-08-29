import numpy as np
import itertools

def get_reward(curr_state, next_state):
  """
  Given the current state and a valid next_state,
  this function returns the reward. 
  Reward = 10, if there are overlapped blue blocks.
  Reward = 0, otherwise.
  """
  q1 = [0,1,2,3]
  q2 = [0,1,4,5]
  q3 = [0,2,4,6]
  num_unique = [q1, q2, q3]
  binary_maps = get_binary(num_unique)
  curr_state = construct_matrixState(curr_state, binary_maps)
  next_state = construct_matrixState(next_state, binary_maps)
  diff_matrix = next_state - curr_state
  _, occupied_cols =  np.where(curr_state==1)
  _, new_occupied_cols =  np.where(diff_matrix==1)
  occupied_cols = set(occupied_cols)
  new_occupied_cols = set(new_occupied_cols)
  reward = 0
  for elem in new_occupied_cols:
    if elem in occupied_cols:
      reward+=10
  return reward

def construct_matrixState(curr_state, block2Binary):
  """
  Construct the binary-matrix representation of a state
  """
  num_ancila = len(curr_state)
  num_data_qubit = 7
  total_qubits = num_data_qubit + num_ancila
  result = np.zeros((num_ancila,total_qubits))
  for i, block in enumerate(curr_state):
    if block != 0:
      row = [0 for j in range(i+1)] + list(block2Binary[block])
      zeros = [0 for j in range(total_qubits - len(row))]
      row = row + zeros
      result[i] = np.array(row)
  return result

def get_binary(num_unique):
  """
  Convert qblock representation to binary representation.
  num_unique is a set that contains qblocks
  """
  block2Binary = {}
  for i, elem in enumerate(num_unique):
    block2Binary[i+1] = np.zeros(7).astype(int)
    for one in elem:
      block2Binary[i+1][one] = 1
  return block2Binary

def get_shift(state_matrix):
  # get the number of shifts
  m1_T = state_matrix.T
  count = 0
  for row in m1_T:
    if np.sum(row) != 0:
      count += 1
  return count

def gen_possStates():
  # generate all the possible states of this particular MDP
  my_list_c = [1,1,2,2,3,3,0,0,0,0,0,0]
  states_space_c = list(itertools.combinations(my_list_c, 6))
  states_space_c = set(states_space_c)
  states = []
  for state in states_space_c:
    temp_state = itertools.permutations(state)
    for one_state in temp_state:
      if one_state not in states:
        states.append(one_state)
  return states

def get_possActions(num_unique_blocks, num_ancila):
  # generate all the possible actions
  actions = np.arange(num_unique_blocks)+1
  final_actions = []
  for action in actions:
    for i in range(num_ancila):
      temp = np.zeros(num_ancila).astype(int)
      temp[i] = action
      final_actions.append(tuple(temp.copy()))
  return final_actions

def p(s,a,s_prime):
    def valid_state(state):
        num_ones = 0; num_twos = 0; num_threes = 0
        for i in state:
            if i==1:
                num_ones += 1
            elif i==2:
                num_twos += 1
            elif i==3:
                num_threes += 1
        if num_ones > 2 or num_twos > 2 or num_threes > 2:
            return False
        return True
    
    diffs = 0
    for i in range(len(s)):
        if s[i] != s_prime[i]:
            diffs+= 1
    if diffs != 1:
        return 0
    
    s_zeros = 0
    s_prime_zeros = 0
    for i in range(len(s)):
        if s[i] == 0:
            s_zeros += 1
        if s_prime[i] == 0:
            s_prime_zeros += 1
    if s_zeros <= s_prime_zeros:
        return 0
    
    if( (not valid_state(s)) or (not valid_state(s_prime))):
        return 0

    new_s = np.array(s) + np.array(a)
    if tuple(new_s) != s_prime:
      return 0

    return 1

def print_state(s):
  q1 = [0,1,2,3]
  q2 = [0,1,4,5]
  q3 = [0,2,4,6]
  num_unique = [q1, q2, q3]
  binary_maps = get_binary(num_unique)
  curr_state = construct_matrixState(s, binary_maps)
  ghostFlag = False
  rowIndex = 0
  for i in curr_state:
    columnIndex = 0
    for j in i:
      if columnIndex <= rowIndex:
         print("🔲", end=" ")
      elif j==0:
          if ghostFlag:
             print("🟩", end=" ")
          else:
            print("　", end=" ")
      else:
          print("🟦", end=" ")
          if sum(i[columnIndex+1:]) > 0:
            ghostFlag = True
          else:
            ghostFlag = False
      columnIndex += 1
    print()
    ghostFlag = False
    rowIndex += 1
  return

states = gen_possStates()
state_to_index = {}
for i, v in enumerate(states):
  state_to_index[v] = i
def take_step(state, action):  
  s_prime = tuple(np.array(state)+ np.array(action))
  if s_prime in state_to_index and p(state, action, s_prime):
    return s_prime
  else:
    return state
  
def repl():
    state = (0,0,0,0,0,0)
    choice = 0
    actions_keyboard = {"a1": (1,0,0,0,0,0), "a2": (0,1,0,0,0,0), "a3": (0,0,1,0,0,0), "a4":(0,0,0,1,0,0), "a5":(0,0,0,0,1,0), "a6":(0,0,0,0,0,1),
                        "b1": (2,0,0,0,0,0), "b2": (0,2,0,0,0,0), "b3": (0,0,2,0,0,0), "b4":(0,0,0,2,0,0), "b5":(0,0,0,0,2,0), "b6":(0,0,0,0,0,2),
                        "c1": (3,0,0,0,0,0), "c2": (0,3,0,0,0,0), "c3": (0,0,3,0,0,0), "c4":(0,0,0,3,0,0), "c5":(0,0,0,0,3,0), "c6":(0,0,0,0,0,3)}
    while choice != "q":
        print_state(state)
        choice = input("Choose an action (a1-6), (b1-6), (c1-6):")
        if choice not in actions_keyboard:
           continue
        new_state = take_step(state, actions_keyboard[choice])
        reward = get_reward(state, new_state)
        print("Reward:", reward)
        state = new_state
            
# q1 = [0,1,2,3]
# q2 = [0,1,4,5]
# q3 = [0,2,4,6]
# num_unique = [q1, q2, q3]

# s1 = (0,0,0,1,0)
# s2 = (0,0,0,1,2)
# print(get_reward(s1,s2))

# states = gen_possStates()
# actions = get_possActions(3, 6)
# print(states[0])

# for s in states:
#     for s_prime in states:
#       for a in actions:
#         if (p(s,a,s_prime) == 1):
#           print(get_reward(s,s_prime))
          # print("Is", s, a, s_prime, "valid?:", p(s,a,s_prime))

if __name__ == "__main__":
  repl()