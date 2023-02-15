#
# '''==================================================
#
# Applying Problem1 on Code1 - Using Larger board of size: 5X5
#
# Problem And Code are both taken FROM:
#
# https://wormtooth.com/20180207-markov-decision-process/
#
# Epsilon: 0.05  ( Bigger then usual, Also Due to the relatively long runtime )
#
# To be used by Shiri Rave, January 2023.
#
# =================================================='''
#
#
# import numpy as np
#
# import sys
# import time
#
# from collections import defaultdict
#
# start = time.time()
# py = 'Python ' + '.'.join(map(str, sys.version_info[:3]))
# print('Jupyter notebook with kernel: {}'.format(py))
#
# # states and actions
# S = [(0, 0), (0, 1), (0, 2), (0, 3),(0, 4),(1, 0), (1, 2),
#      (1, 3),(1, 4), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
#      (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
#      (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
# A = {'U': (-1, 0), 'D': (1, 0), 'R': (0, 1), 'L': (0, -1)}
#
# # reward function
# R = {s: -0.02 for s in S}
# R[(0, 4)] = 1.0
# R[(1, 4)] = -1.0
#
# # transition distributions
# P = defaultdict(int)
# def helper(s, a):
#     h = [(A[a], 0.8)]
#     if a in 'UD':
#         h.append((A['R'], 0.1))
#         h.append((A['L'], 0.1))
#     else:
#         h.append((A['U'], 0.1))
#         h.append((A['D'], 0.1))
#     for d, x in h:
#         n = s[0]+d[0], s[1]+d[1]
#         if n not in S:
#             n = s
#         yield n, x
#
# for s in S:
#     if s in [(0, 4), (1, 4)]:
#         continue
#     for a in A:
#         for n, x in helper(s, a):
#             P[s, a, n] += x
#
# gamma = 0.99
#
#
# # initialize optimal value function
# V = {s: 0.0 for s in S}
# # tolerent error
# error = 0.05
# iters = 1
# while True:
#     nV = {}
#     for s in S:
#         nV[s] = R[s] + gamma * max(sum(P[s, a, n]*V[n] for n in S) for a in A)
#     epsilon = sum(abs(V[s]-nV[s]) for s in S)
#     V = nV
#     iters = iters+1
#     if epsilon < error:
#         break
#
# for s in S:
#     print('{}: {:.2f}'.format(s, V[s]))
#
# # optimal policy
# pi = {s: max(A, key=lambda a: sum(P[s, a, n]*V[n] for n in S))
#      for s in S}
#
# for s in S:
#     if s in [(0, 4), (1, 4)]:
#         continue
#     print('{}: {}'.format(s, pi[s]))
#
# end = time.time()
# diffTime = end - start
# print("Total number of iterations:",iters,", running time:",round(diffTime,5))