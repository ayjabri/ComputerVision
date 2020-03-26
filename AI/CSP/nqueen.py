#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 22:52:50 2020

@author: aymanjabri
"""


class Queen:
    def __init__(self,numQueens):
        self.numQueens = numQueens
        self.board = [[None for i in range(self.numQueens)] for j in range(self.numQueens)]
        
    def placable(self,rowIdx,colIdx):
        if 1 in self.board[rowIdx]:
            return False
        b0 = rowIdx-colIdx
        b1 = rowIdx+colIdx
        for i in range(self.numQueens):
            for j in range(self.numQueens):
                if self.board[i][j]==1 and (i+j==b1 or i-j==b0):
                    return False
        return True
    
    def backTrack(self,colIdx=0):
        if colIdx == self.numQueens: return True
        
        for row in range(self.numQueens):
            if self.placable(row,colIdx):
                self.board[row][colIdx]=1
                if self.backTrack(colIdx+1):
                    return True
                self.board[row][colIdx]=None
        return False
    
    def printQueen(self):
        for i in range(self.numQueens):
            for j in range(self.numQueens):
                if self.board[i][j]==1:
                    print(' Q ',end=''),
                else:
                    print(' - ',end=''),
            print('\n')