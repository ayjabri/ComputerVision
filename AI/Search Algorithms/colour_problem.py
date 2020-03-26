#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:51:08 2020

@author: aymanjabri
"""
import networkx as nx
import matplotlib.pyplot as plt

edges = ([('a', 'c'), ('a', 'd'), ('a', 'b'), ('c', 'f'), ('c', 'b'),
  ('c', 'd'), ('d', 'f'), ('b', 'f'), ('b', 'e'), ('f', 'e')])

grid = nx.Graph()
grid.add_edges_from(edges)
#creates the attribute 'c' in all nodes before solving
for i in grid.nodes:
    grid.nodes[i]['c']=None
    
plt.figure(figsize=(10,10))
pos = nx.spring_layout(grid,seed=0)
nx.draw_networkx(grid,pos= pos, node_size=3000,font_size=18,
                 node_color=('y'))


colours = ['r','y','g']

def valid(graph,colour,node):
    for child in graph[node]:
        if graph.nodes[child]['c']==colour:return False
    return True
def solve(graph,colours,counter): # counter = len(graph.nodes)
    if counter == 0: return True
    for node in graph.nodes:
        if graph.nodes[node]['c']==None:
            for c in colours:
                if valid(graph,c,node):
                    graph.nodes[node]['c']=c
                    if solve(graph,colours,counter-1):
                        return True
                    graph.nodes[node]['c']=None
    return False

def colour(grid):
    node_color = dict(list(grid.nodes('c'))).values()
    plt.figure(figsize=(10,10))
    pos = nx.spring_layout(grid,seed=0)
    nx.draw_networkx(grid,pos= pos, node_size=3000,font_size=18,
                     node_color=node_color)