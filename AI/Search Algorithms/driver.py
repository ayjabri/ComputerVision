#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 09:31:32 2020

@author: ayman jabri
"""



import queue as Q
import time
import resource
import sys
import math
import random


#### SKELETON CODE ####

## Reset Global Variables
path_to_goal=None
cost_of_path=0
nodes_expanded=0
search_depth=0
max_search_depth=0
running_time=0


## The Class that Represents the Puzzle

class PuzzleState(object):
    """docstring for PuzzleState"""
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        if n*n != len(config) or n < 2:
            raise Exception("the length of config is not correct!")
        self.n = n
        self.cost = cost
        self.parent = parent
        self.action = action
        self.dimension = n
        self.config = config
        self.children = {}
        self.blank_index = self.config.index(0)
        for i, item in enumerate(self.config):
            if item == 0:
                self.blank_row = i // self.n
                self.blank_col = i % self.n
                break

    def display(self):
        for i in range(self.n):
            line = []
            offset = i * self.n
            for j in range(self.n):
                line.append(self.config[offset + j])
            print(line)

    def move_left(self):
        if self.blank_col == 0:
            return None
        else:
            target = self.blank_index - 1
            new_config = list(self.config)
            new_config[self.blank_index], new_config[target] = new_config[target], new_config[self.blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):
        if self.blank_col == self.n - 1:
            return None
        else:
            target = self.blank_index + 1
            new_config = list(self.config)
            new_config[self.blank_index], new_config[target] = new_config[target], new_config[self.blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):
        if self.blank_row == 0:
            return None
        else:
            target = self.blank_index - self.n
            new_config = list(self.config)
            new_config[self.blank_index], new_config[target] = new_config[target], new_config[self.blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):
        if self.blank_row == self.n - 1:
            return None
        else:
            target = self.blank_index + self.n
            new_config = list(self.config)
            new_config[self.blank_index], new_config[target] = new_config[target], new_config[self.blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):
        """expand the node"""
        # add child nodes in order of UDLR
        if len(self.children) == 0:
            up_child = self.move_up()
            if up_child is not None:
                self.children['Up']=up_child
            down_child = self.move_down()
            if down_child is not None:
                self.children['Down']=down_child
            left_child = self.move_left()
            if left_child is not None:
                self.children['Left']=left_child
            right_child = self.move_right()
            if right_child is not None:
                self.children['Right']=right_child
        return self.children
    
    def goal(self):
        """create the goal configuration"""
        return tuple(list(range(self.n**2)))

### The function that generates random solvable Puzzle of 'n' size

def CreatePuzzle(n):
    '''Generates solvable puzzle'''
    solvable = False
    model = list(range(n**2))
    while solvable is False:
        random.shuffle(model)
        config = tuple(model)
        p = PuzzleState(config,n)
        solvable = Solvable(p)
    return p

### The function that validates puzzles
       
def Solvable(initial_state):
    config = list(initial_state.config)
    config.remove(0)
    o = 0
    for i in range(8):
        for j in range(i+1,8):
            if config[i]>config[j]: o+=1
    if o%2 != 0:
        return False
    else:
        return True


# Function that Writes to output.txt

def writeOutput():
    file = open('output.txt', 'w')
    file.write("path_to_goal: " + str(path_to_goal))
    file.write("\ncost_of_path: " + str(cost_of_path))
    file.write("\nnodes_expanded: " + str(nodes_expanded))
    file.write("\nsearch_depth: " + str(search_depth))
    file.write("\nmax_search_depth: " + str(max_search_depth))
    file.write("\nrunning_time: " + format(running_time, '.8f'))
    file.write("\nmax_ram_usage: " + format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.0, '.8f'))    
    file.close()
     

### Breadth First Search ###

def bfs_search(initial_state):
    """BFS search"""
    global path_to_goal,cost_of_path,nodes_expanded,search_depth,max_search_depth,running_time
 
    start_time = time.time()
    frontier = Q.deque()
    frontier.append(((),initial_state))
    front_config = set((initial_state.config,))
    explored = set()
    goal = initial_state.goal()
    max_search_depth = 0 
    if initial_state.config == goal: return 'The puzzle is already solved!'
    while frontier:
        sequence,state = frontier.popleft()

        explored.add(state.config)
        children = state.expand()
        if len(sequence)+1 > max_search_depth:
            max_search_depth = len(sequence)+1
        for child_action,child_state in children.items():  
            child_config = child_state.config
            if child_config not in explored and child_config not in front_config:
                new_action = sequence + (child_action,)
                if child_config == goal:
                    path_to_goal = list(new_action)
                    cost_of_path = child_state.cost
                    nodes_expanded = len(explored)
                    search_depth = len(new_action)
                    max_search_depth +=1
                    running_time = time.time()-start_time
                    return writeOutput()
                frontier.append(((new_action),child_state))
                front_config.add((child_config))
    return print('Failure')


### Depth First Search Limited to d = ###
# Children sort is this way to match Columbia's examples
    
def dfs_search(initial_state):
    """DFS search"""
    global path_to_goal,cost_of_path,nodes_expanded,search_depth,max_search_depth,running_time
    
    start_time = time.time()
    frontier = Q.deque()
    frontier.append(((),initial_state))
    front_config = set((initial_state.config,))
    explored = set()
    goal = initial_state.goal()
    if initial_state.config == goal: return 'The puzzle is already solved!'
    
    while frontier:
        sequence,state = frontier.popleft()
        explored.add(state.config)
        try: front_config.remove(state.config) 
        except: pass
#         children = state.expand()
        children = dict(list(state.expand().items())[::-1])
        for child_action,child_state in children.items():
            child_config = child_state.config
            if child_config not in explored and child_config not in front_config:
                new_action = sequence + (child_action,)
                if child_config == goal:
                    path_to_goal = list(new_action)
                    cost_of_path = child_state.cost
                    nodes_expanded = len(explored)
                    search_depth = max_search_depth = len(new_action)
                    running_time = time.time()-start_time
                    return writeOutput()
                frontier.appendleft(((new_action),child_state))
                front_config.add((child_config))
    return 'Failure',len(explored)


### A-star algrithm using Manhattan distance for heuristics ###


def A_star_search(initial_state):

    """A * search"""

    global path_to_goal,cost_of_path,nodes_expanded,search_depth,max_search_depth,running_time

    start_time = time.time()
    n = initial_state.n
    frontier = Q.deque((((),initial_state,0),))
    #frontier.append(((),initial_state,0))
    explored = set()
    goal = initial_state.goal()
    max_search_depth = 0
    if initial_state.config == goal: return 'The puzzle is already solved!'
    
    while frontier:
        sequence,state,fn = frontier.popleft()
        if len(sequence)+1 > max_search_depth:
            max_search_depth = len(sequence)+1
        explored.add(state.config)
        children = state.expand()
        for child_action,child_state in children.items():
            child_config = child_state.config
            if child_config not in explored:
                hn = calculate_manhattan_dist(child_config,n)
                fn = child_state.cost + hn
                
                new_action = sequence + (child_action,)
                if child_config == goal:
                    path_to_goal = list(new_action)
                    cost_of_path = child_state.cost
                    nodes_expanded = len(explored)
                    search_depth = len(new_action)
                    running_time = time.time()-start_time
                    return writeOutput()
                frontier.append(((new_action),child_state,fn))
        frontier = Q.deque(sorted(frontier,key=lambda x:x[2]))

    return 'Failure'


def calculate_total_cost(state):

    """calculate the total estimated cost of a state"""
    return state.cost


def calculate_manhattan_dist(config, n):

    """calculate the manhattan distance of a tile"""
    d = tuple(range(n**2))
    man = sum([abs(config.index(i)%3 - d.index(i)%3) + (abs(config.index(i)//3 - d.index(i)//3)) for i in range(1,n**2)])
    
    return man

def test_goal(puzzle_state,goal):

    """test the state is the goal state or not"""
    
    return puzzle_state.config == goal


# Main Function that reads in Input and Runs corresponding Algorithm

def main():

    sm = sys.argv[1].lower()

    begin_state = sys.argv[2].split(",")

    begin_state = tuple(map(int, begin_state))

    size = int(math.sqrt(len(begin_state)))

    hard_state = PuzzleState(begin_state, size)
    
    if Solvable(hard_state) is False:
        p = CreatPuzzle(3)
        return print('''The entered puzzle is "Unsolvable"
        Refer to: https://en.wikipedia.org/wiki/15_puzzle
        Try this instead {}'''.format(p.config))
    
    if sm == "bfs":

        bfs_search(hard_state)

    elif sm == "dfs":

        dfs_search(hard_state)

    elif sm == "ast":

        A_star_search(hard_state)

    else:

        print("Enter valid command arguments !")

if __name__ == '__main__':

    main()