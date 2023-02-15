# Example : https://wormtooth.com/20180207-markov-decision-process/

import numpy as np
import time
import sys
from collections import defaultdict
py = 'Python ' + '.'.join(map(str, sys.version_info[:3]))
print('Jupyter notebook with kernel: {}'.format(py))


start = time.time()

# states and actions
S = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2),
     (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)]
A = {'U': (-1, 0), 'D': (1, 0), 'R': (0, 1), 'L': (0, -1)}

############################ Applying example 2 on 1 - Without the wall ((1,1) should be a normal state,
# (0,3), (0,2) and (1,2) have special rewards and no actions: ##############################

print ("######## Applying the Second benchmark on the first solution!! ########")

# reward function
R = {s: 0 for s in S}
R[(2, 3)] = 1.0
R[(2, 2)] = -1.0
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


for s in S:
    if s in [(1,2),(2,2),(2,3)]:
        continue
    for a in A:
        for n, x in helper(s, a):
            P[s, a, n] += x

gamma = 0.9


# initialize optimal value function
V = {s: 0.0 for s in S}

# tolerent error
error = 0.005 #0.005 # 0.000003 #

iters = 1
while True:
    nV = {}
    for s in S:
        nV[s] = R[s] + gamma * max(sum(P[s, a, n]*V[n] for n in S) for a in A)
    epsilon = sum(abs(V[s]-nV[s]) for s in S)
    V = nV
    iters = iters + 1
    if epsilon < error:
        break

for s in S:
    print('{}: {:.2f}'.format(s, V[s]))

# optimal policy
pi = {s: max(A, key=lambda a: sum(P[s, a, n]*V[n] for n in S))
     for s in S}

for s in S:
    # if s in [(2, 3),(2,2),(1,2)]:
    #     continue
    print('{}: {}'.format(s, pi[s]))

end = time.time()
diffTime = end - start
print("total number of iterations:",iters,", running time:",diffTime)