

import numpy as np

'''==================================================
Initial set up
=================================================='''

# Hyperparameters
SMALL_ENOUGH = 0.04 #0.005
GAMMA = 0.9
NOISE = 0.10

# Define all states
all_states = []
for i in range(6):
    for j in range(6):
        all_states.append((i, j))

# Define rewards for all states
rewards = {}
for i in all_states:
    if i == (4, 4):
        rewards[i] = -1
    elif i == (5,4):
        rewards[i] = -1
    elif i == (5,5):
        rewards[i] = 1
    else:
        rewards[i] = 0

# Dictionnary of possible actions. We have two "end" states (1,2 and 2,2)
actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('D', 'R', 'L'),
    (0, 2): ('D', 'L', 'R'),
    (0, 3): ('D', 'L', 'R'),
    (0, 4): ('D', 'L', 'R'),
    (0, 5): ('D', 'L'),
    (1, 0): ('D', 'U', 'R'),
    (1, 1): ('D', 'R', 'L', 'U'),
    (1, 2): ('D', 'R', 'L', 'U'),
    (1, 3): ('D', 'R', 'L', 'U'),
    (1, 4): ('D', 'R', 'L', 'U'),
    (1, 5): ('D', 'L', 'U'),
    (2, 0): ('D', 'U', 'R'),
    (2, 1): ('D', 'R', 'L', 'U'),
    (2, 2): ('D', 'R', 'L', 'U'),
    (2, 3): ('D', 'R', 'L', 'U'),
    (2, 4): ('D', 'R', 'L', 'U'),
    (2, 5): ('D', 'L', 'U'),
    (3, 0): ('D', 'U', 'R'),
    (3, 1): ('D', 'R', 'L', 'U'),
    (3, 2): ('D', 'R', 'L', 'U'),
    (3, 3): ('D', 'R', 'L', 'U'),
    (3, 4): ('D', 'R', 'L', 'U'),
    (3, 5): ('D', 'L', 'U'),
    (4, 0): ('D', 'U', 'R'),
    (4, 1): ('D', 'R', 'L', 'U'),
    (4, 2): ('D', 'R', 'L', 'U'),
    (4, 3): ('D', 'R', 'L', 'U'),
    (4, 5): ('D', 'L', 'U'),
    (5, 0): ('U', 'R'),
    (5, 1): ('U', 'L', 'R'),
    (5, 2): ('U', 'L', 'R'),
    (5, 3): ('U', 'L', 'R'),
}

# In the original board: special states do not have actions:
#       (2,2)(->-1), (1,2)(->-1) and (2,3)(->+1)
#  In the wider 6X6 board this should be:
#       (4,4)(->-1), (4,5)(->-1) and (5,5)(->+1)

# Define an initial policy
policy = {}
for s in actions.keys():
    policy[s] = np.random.choice(actions[s])

# Define initial value function
V = {}
for s in all_states:
    if s in actions.keys():
        V[s] = 0
    if s == (4,4):
        V[s] = -1
    if s == (5,4):
        V[s] = -1
    if s == (5,5):
        V[s] = 1

'''==================================================
Value Iteration
=================================================='''

iteration = 0

while True:
    biggest_change = 0
    for s in all_states:
        if s in policy:

            old_v = V[s]
            new_v = 0

            for a in actions[s]:
                if a == 'U':
                    nxt = [s[0] - 1, s[1]]
                if a == 'D':
                    nxt = [s[0] + 1, s[1]]
                if a == 'L':
                    nxt = [s[0], s[1] - 1]
                if a == 'R':
                    nxt = [s[0], s[1] + 1]

                # Choose a new random action to do (transition probability)
                random_1 = np.random.choice([i for i in actions[s] if i != a])
                if random_1 == 'U':
                    act = [s[0] - 1, s[1]]
                if random_1 == 'D':
                    act = [s[0] + 1, s[1]]
                if random_1 == 'L':
                    act = [s[0], s[1] - 1]
                if random_1 == 'R':
                    act = [s[0], s[1] + 1]

                # Calculate the value
                nxt = tuple(nxt)
                act = tuple(act)
                v = rewards[s] + (GAMMA * ((1 - NOISE) * V[nxt] + (NOISE * V[act])))
                if v > new_v:  # Is this the best action so far? If so, keep it
                    new_v = v
                    policy[s] = a

            # Save the best of all actions for the state
            V[s] = new_v
            biggest_change = max(biggest_change, np.abs(old_v - V[s]))

    # See if the loop should stop now
    if biggest_change < SMALL_ENOUGH:
        break
    iteration += 1
    print ("Starting iteration:",iteration)


print("The Final number of iterations is: ", iteration)
print("The Final Resulting values are: ", V)
print("The Final Resulting policy is: ", policy)