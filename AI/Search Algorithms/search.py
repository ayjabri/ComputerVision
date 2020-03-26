#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:06:48 2020

@author: aymanjabri
"""
#from queue import PriorityQueue
from graph import Graph
from PIL import Image


# =============================================================================
# Examples
# =============================================================================
img = Image.open('img.png')

g = Graph()
for i in 'ABCDEFGHIS':
    g.addNode(i)

g.connect('S','A',6)
g.connect('S','B',2)
g.connect('S','C',5)
g.connect('A','D',9)
g.connect('B','E',3)
g.connect('C','H',2)
g.connect('E','A',2)
g.connect('H','G',7)
g.connect('H','F',2)
g.connect('F','D',4)
g.connect('G','E',5)
g.connect('G','D',1)
# g.connect('S','H',0)
# g.connect('A','F',8)
# g.connect('H','D',1)


################ USA Map ###################
us = Graph()
cities = ['Montreal','Boston','New York','Washington','Pittsburgh','Toronto',
          'Sault Ste Marie','Chicago','Raleigh','Saint Louis','Omaha','Duluth',
          'Winnipeg','Charleston','Atlanta','Nashville','Miami','Little Rock',
          'New Orleans','Houston','Dallas','Oklahoma City',
          'Kansas City','Denver','Santa Fe','El Paso','Phoenix','Los Angeles',
          'Las Vegas','Salt Lake City','Helena','Calgary','Vancouver','Seattle',
          'Portland','San Francisco']
for city in cities:
    us.addNode(city)

links = [('Montreal','Boston',69),('Montreal','New York',99),('Montreal','Toronto',115),
 ('Montreal','Sault Ste Marie',193),('Boston','New York',74),('New York','Washington',76),
 ('New York','Pittsburgh',69),('Pittsburgh','Toronto',80),('Pittsburgh','Washington',85),
 ('Toronto','Sault Ste Marie',90),('Charleston','Atlanta',63),
 ('Sault Ste Marie','Winnipeg',156),('Sault Ste Marie','Duluth',110),
 ('Duluth','Omaha',74),('Duluth','Chicago',157),('Chicago','Pittsburgh',81),
 ('Chicago','Saint Louis',104),('Raleigh','Washington',47),('Raleigh','Charleston',95),
 ('Raleigh','Atlanta',96),('Raleigh','Nashville',128),('Charleston','Miami',80),
 ('Atlanta','Miami',116),('Atlanta','New Orleans',120),('Miami','New Orleans',151),
 ('New Orleans','Little Rock',100),('New Orleans','Houston',80),('Houston','Dallas',46),
 ('Saint Louis','Little Rock',60),('Saint Louis','Nashville',85),
 ('Saint Louis','Kansas City',68),('Chicago','Omaha',142),('Nashville','Little Rock',94),
 ('Little Rock','Dallas',74),('Little Rock','Oklahoma City',72),('Omaha','Denver',130),
 ('Omaha','Helena',174),('Helena','Duluth',150),('Helena','Winnipeg',137),
 ('Helena','Calgary',130),('Helena','Seattle',189),('Helena','Salt Lake City',116),
 ('Helena','Denver',126),('Denver','Salt Lake City',101),('Denver','Phoenix',128),
 ('Denver','Santa Fe',70),('Denver','Kansas City',135),('Salt Lake City','Las Vegas',89),
 ('Salt Lake City','San Francisco',156),('Salt Lake City','Portland',175),
 ('Winnipeg','Duluth',103),('Winnipeg','Calgary',180),('Calgary','Vancouver',100),
 ('Calgary','Seattle',118),('Seattle','Vancouver',45),('Seattle','Portland',44),
 ('Portland','San Francisco',151),('San Francisco','Los Angeles',100),
 ('Los Angeles','Las Vegas',66),('Los Angeles','Phoenix',109),('Los Angeles','El Paso',191),
 ('Phoenix','Santa Fe',85),('Santa Fe','Oklahoma City',121),('Santa Fe','El Paso',65),
 ('El Paso','Dallas',140),('Kansas City','Oklahoma City',135)]

for i,j,c in links:
    us.connect(i,j,c)
    us.connect(j,i,c)

################ Tree ###################

tree = Graph()
for i in 'ABCDEFGHIJKLMNO':
    tree.addNode(i)
tree_links =[('A','B',1),('A','C',1),('B','D',1),('B','E',1),('C','F',1),('C','G',1),
             ('D','H',1),('D','I',1),('E','J',1),('E','K',1),('F','L',1),('F','M',1),
             ('G','N',1),('G','O',1)]
for i,j,c in tree_links:
    tree.connect(i,j,c)
    tree.connect(j,i,c)
    
# tree = {'A':['B','C'],'B':['A','D','E'],'C':['A','F','G'],
#          'D':['B','H','I'],'E':['B','J','K'],'F':['C','L','M'],
#          'G':['C','N','O'],'H':'D','I':'D','J':'E','K':'E','L':'F',
#          'M':'F','N':'G','O':'G'}

from collections import deque
# =============================================================================
# AI Search Agent using BFS,DFS and UCS algorithms
# =============================================================================
class SearchAgent(object):
    def __init__(self,graph):
        self.graph=graph
    
    def _error(self,initialState,goalTest):
        return 'There is no path between {} & {}'.format(initialState,goalTest)
        
    def bfs_explored(self,initialState,goalTest):
        frontier = deque(initialState)
        explored = deque()
        while frontier:
            state = frontier.popleft()
            explored.append(state)
            
            if state==goalTest:
                return print('Success\nExplored nodes {}'.format(list(explored)))
            
            frontier.extend(i for i in self.graph[state] if i not in (frontier+explored))
            ## Another longe way to add nodes to frontier 
            # neighbours = self.graph[state]
            # for n in sorted(neighbours):
            #     if n not in (explored+frontier):
            #         frontier.append(n)
        return self._error(initialState,goalTest)
    
    def BFS_path(self,initialState,goalTest):
        frontier=[[initialState]]
        explored=[]
        while frontier:
            path = frontier.pop(0)
            state = path[-1]
            explored.append(state)
            if state==goalTest:return print('BFS path is:{}\nVisited:{}'.format(
                    path,explored))
            neighbours=self.graph.getSuccessors(state)
            for n in sorted(neighbours):
                exist = [i[-1] for i in frontier]
                if n not in (explored+exist):
                    new_path = path.copy()
                    new_path.append(n)
                    # if n==goalTest:
                    #     return print('The shortest path between {} & {} is:\n{}'.format(
                    # initialState,goalTest,new_path))
                    frontier.append(new_path)
        return self._error(initialState,goalTest)
    
    def dfs_explored(self,initialState,goalTest):
        frontier=deque(initialState)
        explored=deque()
        
        while frontier:
            state = frontier.pop()
            explored.append(state)
            if state==goalTest:
                return print('Success\nExplored nodes {}'.format(list(explored)))
            frontier.extend(i for i in sorted(self.graph[state],reverse=True
                                          ) if i not in (frontier+explored))
        return self._error(initialState,goalTest)
    
    def DFS_path(self,initialState,goalTest):
        frontier = [[initialState]]
        explored = []
        while frontier:
            path = frontier.pop()
            state = path[-1]
            explored.append(state)
            neighbours = self.graph.getSuccessors(state)
            for n in sorted(neighbours,reverse=True):
                if n not in explored:
                    new_path = path.copy()
                    new_path.append(n)
                    if n==goalTest:
                        explored.append(goalTest)
                        return print('DFS path is:{}\nVisited Nodes {}'.format(
                    new_path,explored))
                    frontier.append(new_path)
        return self._error(initialState,goalTest)
    
    def UCSearch(self,initialState,goalTest):
        frontier = {initialState:0}
        visited =[]
        while frontier:
           state,cost = frontier.popitem()
           visited.append(state)
           if state==goalTest: return 'Success {}'.format(visited)
           neighbours = self.graph.getSuccessors(state)
           for n in neighbours:
               f = self.graph.getWeightEdge(state,n)
               if n not in visited and n not in frontier:
                   frontier[n]=cost+f
                   frontier = dict(sorted(frontier.items(),key=lambda x:x[1],
                                                  reverse=True))
               elif n in frontier and frontier[n]>cost+f:
                   frontier[n]=cost

        return self._error(initialState,goalTest)

    
    def UCSearch_path(self,start,end):
        if (start not in self.graph.nodes or end not in self.graph.nodes):
            return 'One of the entered points is not in Graph'
        frontier = {(start,):0}
        explored = []
        while frontier:
            path,cost = frontier.popitem() #{'S': 0, 'SA': 2}
            node = path[-1]
            if node==end: 
                explored.append(node)
                return 'Path {} Cost {}'.format(path,cost)
            neighbours = self.graph.getSuccessors(node)
            if node not in explored:
                for n in neighbours:
                    f = self.graph.getWeightEdge(node,n)
                    new_path,new_cost = path+(n,),f+cost
                    exist = [i for i in frontier.keys() if i[-1]==n]
                    if len(exist)>0 and frontier[exist[0]]>new_cost:
                        frontier.pop(exist[0])
                    frontier[new_path]=new_cost
                explored.append(node)
            frontier = dict(sorted(frontier.items(),key=lambda x:x[1],reverse=True))
        return self._error(start,end)