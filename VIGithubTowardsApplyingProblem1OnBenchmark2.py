

import numpy as np

'''==================================================
Initial set up

https://towardsdatascience.com/how-to-code-the-value-iteration-algorithm-for-reinforcement-learning-8fb806e117d1

=================================================='''
import time

start = time.time()
# Hyperparameters
SMALL_ENOUGH = 0.005 # 0.000003 #
GAMMA = 0.9
NOISE = 0.10

# Define all states
all_states = []
for i in range(3):
    for j in range(4):
        all_states.append((i, j))

# states - first benchmark:

S = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 2),
     (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)]

# Define rewards for all states
rewards = {}
for i in all_states:
    if i == (0, 3):
        rewards[i] = 1
    elif i == (1, 3):
        rewards[i] = -1
    elif i == (2, 3):
        rewards[i] = -1
    else:
        rewards[i] = -0.02

# actions - first benchmark:

# A = {'U': (-1, 0), 'D': (1, 0), 'R': (0, 1), 'L': (0, -1)}

# Dictionary of possible actions. We have two "end" states (1,2 and 2,2)
# --> Need to fix this.
actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('R', 'L'),
    (0, 2): ('D', 'L', 'R'),
    #(0, 3): ('D', 'L'),
    (1, 0): ('D', 'U'),
    # (1, 1): ('D', 'R', 'L', 'U'), --> (1,1) should not exist.
    (1, 2): ('D', 'U', 'R'),
    (1, 3): ('D', 'L', 'U'),
    (2, 0): ('U', 'R'),
    (2, 1): ('L', 'R'),
    (2, 2): ('L', 'R', 'U'),
    (2, 3): ('L', 'U')
}

# Define an initial policy
policy = {}
for s in actions.keys():
    policy[s] = np.random.choice(actions[s])

# Define initial value function
V = {}

#  reward function - first benchmark:
# R = {s: -0.02 for s in S}
# R[(0, 3)] = 1.0
# R[(1, 3)] = -1.0
for s in all_states:
    if s in actions.keys():
        V[s] = -0.02
    if s == (0, 3):
        V[s] = 1
    if s == (1, 3):
        V[s] = -1

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


end = time.time()
diffTime = end - start
print("total number of iterations:",iteration,", running time:",diffTime)
print("The Final number of iterations is: ", iteration)
print("The Final Resulting values are: ", V)
print("The Final Resulting policy is: ", policy)