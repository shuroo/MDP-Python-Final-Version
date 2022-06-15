# Example : https://wormtooth.com/20180207-markov-decision-process/

import numpy as np

import sys
from collections import defaultdict
py = 'Python ' + '.'.join(map(str, sys.version_info[:3]))
print('Jupyter notebook with kernel: {}'.format(py))

# states and actions
S = [(0, 0), (0, 1), (0, 2), (0, 3),(0, 4),(0, 5), (1, 0), (1, 2),
     (1, 3),(1, 4),(1, 5), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
     (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
A = {'U': (-1, 0), 'D': (1, 0), 'R': (0, 1), 'L': (0, -1)}

# reward function
R = {s: -0.02 for s in S}
R[(0, 5)] = 1.0
R[(1, 5)] = -1.0

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
    if s in [(0, 5), (1, 5)]:
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
    if s in [(0, 5), (1, 5)]:
        continue
    print('{}: {}'.format(s, pi[s]))
