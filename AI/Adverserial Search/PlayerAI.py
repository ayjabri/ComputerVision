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

