'''==================================================

Applying Problem2 on Code1  -

The Code is taken FROM:

https://wormtooth.com/20180207-markov-decision-process/

The problem is taken from:

https://towardsdatascience.com/how-to-code-the-value-iteration-algorithm-for-reinforcement-learning-8fb806e117d1

Epsilon: error = 3*10**(-5)

To be used by Shiri Rave, January 2023.

=================================================='''

import numpy as np
import time

import sys
from collections import defaultdict
py = 'Python ' + '.'.join(map(str, sys.version_info[:3]))
print('Jupyter notebook with kernel: {}'.format(py))

start = time.time()

# states and actions
S = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0),(1, 1), (1, 2),
     (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)]
A = {'U': (-1, 0), 'D': (1, 0), 'R': (0, 1), 'L': (0, -1)}


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
        random_action = np.random.choice(['R','L'])
        h.append((A[random_action], 0.1))
    else:
        random_action = np.random.choice(['U','D'])
        h.append((A[random_action], 0.1))
    for d, x in h:
        n = s[0]+d[0], s[1]+d[1]
        if n not in S:
            n = s
        yield n, x


# initialize optimal value function
V = {s: 0.0 for s in S}
iteration = 1

############################ Applying example 2 on 1 ##############################

print ("######## Applying the Second benchmark on the first solution!! ########")

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
error = 0.0000003

while True:
    nV = {}
    for s in S:
        nV[s] = R[s] + gamma * max(sum(P[s, a, n]*V[n] for n in S) for a in A)
    epsilon = sum(abs(V[s]-nV[s]) for s in S)
    V = nV
    iteration = iteration + 1
    if epsilon < error:
        break

for s in S:
    print('{}: {:.2f}'.format(s, V[s]))

# optimal policy
pi = {s: max(A, key=lambda a: sum(P[s, a, n]*V[n] for n in S))
     for s in S}

for s in S:
    print('{}: {}'.format(s, pi[s]))


end = time.time()
diffTime = end - start
print("Total number of iterations:",iteration,", running time:",diffTime)