#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:59:44 2020

@author: ayman jabri
"""

import random
from BaseAI import BaseAI
import time
from math import inf
import math
from numpy import argmax
import numpy as np
from functools import lru_cache

# =============================================================================
# This Section is to define the player
# =============================================================================
 

class PlayerAI(BaseAI):

    def getMove(self, grid):        
        moves = grid.getAvailableMoves()  
        
        return self.decision(grid) if moves else None
    
    def decision(self,grid): #add depth here to experement
        self.start_time = time.process_time()
        self.depth = 7
        maxUtility = []
        moves = grid.getAvailableMoves()
        for move in moves:
            c = grid.clone()
            c.move(move)
            maxUtility.append(self.MiniMax(c,self.depth))
        
        return moves[argmax(maxUtility)],time.process_time()-self.start_time
    
    @lru_cache(maxsize=1000)
    def MiniMax(self,grid,depth,maxPlayer=True):
        if (depth==0 or time.process_time()-self.start_time>=0.199 or grid.getMaxTile()==2048):
            return self.Heuristics(grid)
        if maxPlayer:
            v = - inf
            for child in self.children(grid):
                v = max(v,self.MiniMax(child,depth-1),False)
            return v
        else: #Min function: all possible tiles 
            avail = grid.getAvailableCells()
            pos = avail[random.randint(0,len(avail)-1)]
            grid.insertTile(pos,self.insertRandom())
            v = self.MiniMax(grid,depth-1,True)
            return v

    def insertRandom(self):
        if random.random()>=0.9: return 4
        return 2
    
    @staticmethod
    def children(grid):
        children = []
        for m in grid.getAvailableMoves():
            c = grid.clone()
            c.move(m)
            children.append(c)
        return children

    @staticmethod
    def Heuristics(grid):
        #Heuristic the number of free tiles plus max tile
        a = np.array(grid.map)
        h1 = len(grid.getAvailableCells())
        h2 = math.log2(grid.getMaxTile())
        h3 = (10 if (a.argmax()%4,a.argmax()//4) in ((0,0),(0,3),(3,0),(3,3)) else 0)
        smoothness = 0
        for row in range(4):
            for column in range(4):
                s = inf
                if row > 0:
                    s = min(s, abs((grid.map[row][column] or 2) - (grid.map[row - 1][column] or 2)))
                if column > 0:
                    s = min(s, abs((grid.map[row][column] or 2) - (grid.map[row][column - 1] or 2)))
                if row < 3:
                    s = min(s, abs((grid.map[row][column] or 2) - (grid.map[row + 1][column] or 2)))
                if column < 3:
                    s = min(s, abs((grid.map[row][column] or 2) - (grid.map[row][column + 1] or 2)))

                smoothness -= s

        h4 = smoothness
        
        #count of 2's and 4's
        h5 = (a==2).sum()+(a==4).sum()
        h6 = monotonicity(grid)
        w1,w2,w3,w4,w5,w6 = 1,1,1,1,-1,1
        h = w1*h1 + w2*h2 + w3*h3 + w4*h4 + w5*h5 + h6*w6
        return h 




# from random import randint
# from BaseAI import BaseAI
# import math
# import time

# start = 0


# class PlayerAI(BaseAI):
#     def __init__(self):
#         self.possibleNewTiles = [2, 4]
#         self.probability = 0.9
#         self.time_limit = 0.1

#         self.smoothnessWeight = 0.1
#         self.monotonicityWeight = 1.0
#         self.emptyWeight = 2.7
#         self.maxWeight = 1.0
#         self.distanceWeight = 10.0

#     def evaluate(self, grid):

#         empty_cells = len(grid.getAvailableCells())
#         max_tile = grid.getMaxTile()

#         smoothness = self.smoothness(grid) * self.smoothnessWeight
#         monotonicity = self.monotonicity(grid) * self.monotonicityWeight
#         emptiness = (math.log(empty_cells) / math.log(2)) * self.emptyWeight if empty_cells != 0 else 0
#         maxvalue = self.get_max_value(max_tile, empty_cells) * self.maxWeight
#         distance = self.distance(grid, max_tile) * self.distanceWeight

#         return emptiness + monotonicity + smoothness + maxvalue + distance

#     @staticmethod
#     def distance(grid, max_tile):
#         dis = None

#         for x in range(grid.size):

#             if dis:
#                 break

#             for y in range(grid.size):
#                 if max_tile == grid.map[x][y]:

#                     if max_tile < 1024:
#                         dis = -((abs(x - 0) + abs(y - 0)) * max_tile)
#                     else:
#                         dis = -((abs(x - 0) + abs(y - 0)) * (max_tile / 2))
#                     break

#         return dis

#     @staticmethod
#     def get_max_value(max_tile, empty_cells):
#         return math.log(max_tile) * empty_cells / math.log(2)


def monotonicity(grid):

    totals = [0, 0, 0, 0]

    for x in range(3):

        currentIndex = 0
        nextIndex = currentIndex + 1

        while nextIndex < 4:
            while nextIndex < 4 and grid.map[x][nextIndex] == 0:
                nextIndex += 1

            if nextIndex >= 4:
                nextIndex -= 1

            currentValue = math.log(grid.map[x][currentIndex]) / math.log(2) if grid.map[x][currentIndex] else 0
            nextValue = math.log(grid.map[x][nextIndex]) / math.log(2) if grid.map[x][nextIndex] else 0

            if currentValue > nextValue:
                totals[0] += currentValue + nextValue
            elif nextValue > currentValue:
                totals[1] += currentValue - nextValue

            currentIndex = nextIndex
            nextIndex += 1

    for y in range(3):

        currentIndex = 0
        nextIndex = currentIndex + 1

        while nextIndex < 4:
            while nextIndex < 4 and grid.map[nextIndex][y] == 0:
                nextIndex += 1

            if nextIndex >= 4:
                nextIndex -= 1

            currentValue = math.log(grid.map[currentIndex][y]) / math.log(2) if grid.map[currentIndex][y] else 0
            nextValue = math.log(grid.map[nextIndex][y]) / math.log(2) if grid.map[nextIndex][y] else 0

            if currentValue > nextValue:
                totals[2] += nextValue - currentValue
            elif nextValue > currentValue:
                totals[3] += currentValue - nextValue

            currentIndex = nextIndex
            nextIndex += 1

    return max(totals[0], totals[1]) + max(totals[2], totals[3])

#     @staticmethod
#     def smoothness(grid):

#         smoothness = 0

#         for x in range(grid.size):
#             for y in range(grid.size):
#                 s = float('infinity')

#                 if x > 0:
#                     s = min(s, abs((grid.map[x][y] or 2) - (grid.map[x - 1][y] or 2)))
#                 if y > 0:
#                     s = min(s, abs((grid.map[x][y] or 2) - (grid.map[x][y - 1] or 2)))
#                 if x < 3:
#                     s = min(s, abs((grid.map[x][y] or 2) - (grid.map[x + 1][y] or 2)))
#                 if y < 3:
#                     s = min(s, abs((grid.map[x][y] or 2) - (grid.map[x][y + 1] or 2)))

#                 smoothness -= s

#         return smoothness

#     def get_new_tile(self):
#         if randint(0, 99) < 100 * self.probability:
#             return self.possibleNewTiles[0]
#         else:
#             return self.possibleNewTiles[1]

#     def search(self, grid, alpha, beta, depth, player):

#         if time.clock() - start > self.time_limit:
#             return self.evaluate(grid), -1, True

#         if depth == 0:
#             return self.evaluate(grid), -1, False

#         if player:

#             best_score, best_move = alpha, None

#             positions = grid.getAvailableMoves()

#             if len(positions) == 0:
#                 return self.evaluate(grid), None, False

#             for position in positions:

#                 new_grid = grid.clone()
#                 new_grid.move(position)

#                 score, move, timeout = self.search(new_grid, alpha, beta, depth - 1, False)

#                 if score > best_score:
#                     best_score, best_move = score, position

#                 if best_score >= beta:
#                     break

#                 if best_score > alpha:
#                     alpha = best_score

#             return best_score, best_move, False

#         else:

#             best_score, best_move = beta, None

#             cells = grid.getAvailableCells()

#             if len(cells) == 0:
#                 return self.evaluate(grid), None, False

#             for cell in cells:

#                 value = self.get_new_tile()

#                 new_grid = grid.clone()
#                 new_grid.setCellValue(cell, value)

#                 score, move, timeout = self.search(new_grid, alpha, beta, depth - 1, True)

#                 if score < best_score:
#                     best_score, best_move = score, None

#                 if best_score <= alpha:
#                     break

#                 if best_score < beta:
#                     beta = best_score

#             return best_score, None, False

#     def iterative(self, grid):
#         global start
#         best_score, depth = -float('infinity'), 1

#         start = time.clock()

#         while True:

#             score, move, timeout = self.search(grid, -float('infinity'), float('infinity'), depth, True)

#             if timeout:
#                 break

#             if score > best_score:
#                 best_move, best_score = move, score

#             depth += 1

#         return best_move

#     def getMove(self, grid):
#         return self.iterative(grid)