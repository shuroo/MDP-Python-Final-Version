# Example : https://wormtooth.com/20180207-markov-decision-process/

import numpy as np

import sys
from collections import defaultdict
py = 'Python ' + '.'.join(map(str, sys.version_info[:3]))
print('Jupyter notebook with kernel: {}'.format(py))

# states and actions
S = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 2),
     (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)]
A = {'U': (-1, 0), 'D': (1, 0), 'R': (0, 1), 'L': (0, -1)}

# reward function
R = {s: -0.02 for s in S}
R[(0, 3)] = 1.0
R[(1, 3)] = -1.0

# transition distributions
P = defaultdict(int)
def helper(s, a):
    h = [(A[a], 0.8)]
    if a in 'UD':
        h.append((A['R'], 0.1))
        h.append((A['L'], 0.1))
    else:
        h.append((A['U'], 0.1))
        h.append((A['D'], 0.1))
    for d, x in h:
        n = s[0]+d[0], s[1]+d[1]
        if n not in S:
            n = s
        yield n, x

for s in S:
    if s in [(0, 3), (1, 3)]:
        continue
    for a in A:
        for n, x in helper(s, a):
            P[s, a, n] += x

gamma = 0.99


# initialize optimal value function
V = {s: 0.0 for s in S}

# tolerent error
error = 10**(-3)

while True:
    nV = {}
    for s in S:
        nV[s] = R[s] + gamma * max(sum(P[s, a, n]*V[n] for n in S) for a in A)
    epsilon = sum(abs(V[s]-nV[s]) for s in S)
    V = nV
    if epsilon < error:
        break

for s in S:
    print('{}: {:.2f}'.format(s, V[s]))

# optimal policy
pi = {s: max(A, key=lambda a: sum(P[s, a, n]*V[n] for n in S))
     for s in S}

for s in S:
    if s in [(0, 3), (1, 3)]:
        continue
    print('{}: {}'.format(s, pi[s]))

# # policy pi
# pi = {(0, 0): 'R', (0, 1): 'R', (0, 2): 'R', (0, 3): '#',
#       (1, 0): 'D', (1, 1): '#', (1, 2): 'R', (1, 3): '#',
#       (2, 0): 'R', (2, 1): 'R', (2, 2): 'U', (2, 3): 'U'}
#
# # calculate value function for pi
# # with tolerent error
# def val_fn(pi, error=10**(-3)):
#     V = {s: 0.0 for s in S}
#
#     while True:
#         nV = {}
#         for s in S:
#             nV[s] = R[s] + gamma * sum(P[s, pi[s], n]*V[n] for n in S)
#         epsilon = sum(abs(V[s]-nV[s]) for s in S)
#         V = nV
#         if epsilon < error:
#             break
#     return V
#
# V = val_fn(pi)
# for s in S:
#     print('{}: {:.2f}'.format(s, V[s]))


############################ Applying example 2 on 1 ##############################

print ("######## Applying the Second benchmark on the first solution!! ########")
# Hyperparameters
# SMALL_ENOUGH = 0.005
# GAMMA = 0.9
# NOISE = 0.10

# Define rewards for all states
# rewards = {}
# for i in all_states:
#     if i == (1, 2):
#         rewards[i] = -1
#     elif i == (2, 2):
#         rewards[i] = -1
#     elif i == (2, 3):
#         rewards[i] = 1
#     else:
#         rewards[i] = 0

# reward function
R = {s: 0 for s in S}
R[(0, 3)] = 1.0
R[(0, 2)] = -1.0
R[(1, 2)] = -1.0

# transition distributions
P = defaultdict(int)
def helper(s, a):
    h = [(A[a], 0.9)]
    if a in 'UD':
        # h.append((A['R'], 0.1))
        #h.append((A['L'], 0.1))
        random_action = np.random.choice(['R','L'])
        h.append((A[random_action], 0.1))
    else:
        #h.append((A['U'], 0.1))
        #h.append((A['D'], 0.1))
        random_action = np.random.choice(['U','D'])
        h.append((A[random_action], 0.1))
    for d, x in h:
        n = s[0]+d[0], s[1]+d[1]
        if n not in S:
            n = s
        yield n, x

for s in S:
    if s in [(0,3),(0,2),(1,2)]:
        continue
    for a in A:
        for n, x in helper(s, a):
            P[s, a, n] += x

gamma = 0.9


# initialize optimal value function
V = {s: 0.0 for s in S}

# tolerent error
error = 0.000003 #0.005

while True:
    nV = {}
    for s in S:
        nV[s] = R[s] + gamma * max(sum(P[s, a, n]*V[n] for n in S) for a in A)
    epsilon = sum(abs(V[s]-nV[s]) for s in S)
    V = nV
    if epsilon < error:
        break

for s in S:
    print('{}: {:.2f}'.format(s, V[s]))

# optimal policy
pi = {s: max(A, key=lambda a: sum(P[s, a, n]*V[n] for n in S))
     for s in S}

for s in S:
    # if s in [(2, 3),(0,2),(1,2)]:
    #     continue
    print('{}: {}'.format(s, pi[s]))

