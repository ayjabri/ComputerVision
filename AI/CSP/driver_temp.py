#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 06:38:00 2020

@author: aymanjabri
"""


import numpy as np
import sys

global possible_values


class Sudoku(object):
    def __init__(self,sudoku):
        self.grid = np.array(list(sudoku),dtype=int).reshape(9,9)
        self.solution = np.zeros((9,9))
            
    def Possible(self,row,col,n):
        i = row//3 * 3
        j = col//3 * 3
        if (n in self.grid[i:i+3,j:j+3]
            or n in self.grid[row,:] 
            or n in self.grid[:,col]):
            return False
        return True
    
    def solve(self):
        for i in range(9):
            for j in range(9):
                if self.grid[i,j]==0:
                    for n in range(1,10):
                        if self.Possible(i,j,n):
                            self.grid[i,j]=n
                            self.solve()
                            self.grid[i,j]=0
                    return
        self.solution = np.array(self.grid)
        
        # return False
        # print(self.grid)
        # input('More?')
        
    def solve_test(self):
            for i in range(9):
                for j in range(9):
                    if self.grid[i,j]==0:
                        for n in possible_values[i,j]:
                            if self.Possible(i,j,n):
                                self.grid[i,j]=n
                                self.solve()
                                possible_values[i,j].remove(n)
                                self.grid[i,j]=0
                        return
            self.solution = np.array(self.grid)
            
        
#define the domain of possible values 
def Domains(suduko):
    possible_values = {}
    for i in range(9):
        for j in range(9):
            if suduko[i,j] != 0:
                possible_values[i,j]= [suduko[i,j]]
            else:
                possible_values[i,j] = list(m for m in range(1,10) if m not in suduko[:,i] and m not in suduko[j,:])
    return possible_values

# g = Sudoku('000007000090001000000045006000020000036000410500000809000000004000018000081500032')


# def main():
    
#     # puzzle = sys.argv[1]
#     with open('sudokus_start.txt','r') as p:
#         puzzles = p.readlines()
    
#     for puzzle in puzzles:
#         grid = np.array(list(puzzle.strip()),dtype=int).reshape(9,9)
#         solve(grid)
#         # print(grid.solution)
#         with open('output.txt','a') as file:
#             file.write(''.join([str(elem) for elem in grid.reshape(81)]))
#             file.write(' BTS\n')
    
    
# if __name__ =='__main__':
#     main()