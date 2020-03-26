#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 06:42:37 2020

@author: aymanjabri
"""
import matplotlib.pyplot as plt
from random import randint
import networkx as nx
from math import inf
import time

tree = nx.balanced_tree(4,10,create_using=nx.DiGraph())

for i in tree.nodes:
    tree.nodes[i]['h']=randint(-100,100)

# print(tree.nodes('h'))

def Decision(graph,node,depth):
    nextDir = -1
    maxUtility = -inf
    moves = nx.neighbors(graph,node)
    for move in moves:
        utility = minimax(graph,move,depth,False)
        if utility >= maxUtility:
            maxUtility=utility
            nextDir = move
    return nextDir
    
def minimax(graph,node,depth,maxPlay=True):
    if depth==0: return graph.nodes[node]['h']
    if maxPlay:
        v = - inf
        for i in graph[node]:
            v = max(v,minimax(graph,i,depth-1,maxPlay=False))
        return v
    else:
        v = inf
        for j in graph[node]:
            v = min(v,minimax(graph,j,depth-1,maxPlay=True))
        return v
    
def MinMax(graph,node,depth,maxplayer=True,time=0.2):
    if depth==0 or graph.out_degree(node)==0:return graph.nodes[node]['h']
    if maxplayer:
        utility = -inf
        for n in graph[node]:
            utility = max(utility,MinMax(graph,n,depth-1,maxplayer=False))
        return utility
    else:
        utility = inf
        for n in graph[node]:
            utility = min(utility,MinMax(graph,n,depth-1,maxplayer=True))
        return utility
    
            

# l = dict(tree.nodes('h'))
# plt.figure(figsize=(14,10))
# top = nx.bipartite.sets(tree)[0]
# pos = nx.kamada_kawai_layout(tree,scale=0.25,center=[1,0])
# nx.draw_networkx(tree,pos=pos,with_labels=False,font_size=24,node_size=1500)
# nx.draw_networkx_labels(tree,pos=pos,labels=l)


