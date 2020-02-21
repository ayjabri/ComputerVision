#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:50:58 2020

@author: aymanjabri
"""
import networkx as nx
import matplotlib.pyplot as plt

# =============================================================================
# US map as mentioned in Columbia AI course
# =============================================================================
cities = [('Montreal','Boston',69),('Montreal','New York',99),('Montreal','Toronto',115),
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

us = nx.Graph()
us.add_weighted_edges_from(cities)

# =============================================================================
# DiGraph used in Columbia AI course
# =============================================================================

class Simple(nx.DiGraph):
    def __init__(self):
        
        nodes=[('S','A',6),('S','B',2),('S','C',5),('B','E',3),('A','D',9),
            ('C','H',2),('E','A',2),('H','F',2),('H','G',7),('G','D',1),
            ('F','D',4),('G','E',5)]
        self.add_weighted_edges_from(nodes)
        self.pos = {'S':([0., 0.]),
                    'A': ([1.,0.]),
                    'B': ([0.,-1.]),
                    'C': ([0., 1.]),
                    'E': ([1.,-1]),
                    'D': ([2., 0.]),
                    'H': ([1.,1.]),
                    'F': ([2.,1.]),
                    'G': ([2.,-1.])}
    def plot(self):
        self.weights = nx.get_edge_attributes(self,'weight')
        plt.figure(figsize=(8,8))
        nx.draw_networkx_edge_labels(self,self.pos,edge_labels=self.weights,font_size=18)
        nx.draw(self,pos=self.pos,with_labels=True,node_size=5000,font_size=24,width=3,
                arrow_size=40,node_color='C9')
        
# n =nx.DiGraph()
# n.add_weighted_edges_from(nodes)

# p = {'S':([0., 0.]),
#  'A': ([1.,0.]),
#  'B': ([0.,-1.]),
#  'C': ([0., 1.]),
#  'E': ([1.,-1]),
#  'D': ([2., 0.]),
#  'H': ([1.,1.]),
#  'F': ([2.,1.]),
#  'G': ([2.,-1.])}

# l = nx.get_edge_attributes(n,'weight')

# plt.figure(figsize=(8,8))
# nx.draw_networkx_edge_labels(n,pos=p,edge_labels=l,font_size=18)
# nx.draw(n,pos=p,with_labels=True,node_size=5000,font_size=24,width=3,
#         arrow_size=40,node_color='C9')

# =============================================================================
# Romania
# =============================================================================

ro = nx.Graph()
ro_cities=[('Arad','Sibiu',140),
('Arad','Timisoara',118),
('Arad','Zerind',75),
('Bucharest','Fagaras',211),
('Bucharest','Giurgiu',90),
('Bucharest','Pitesti',101),
('Bucharest','Urziceni',85),
('Craiova','Dobreta',120),
('Craiova','Pitesti',138),
('Craiova','Rimnicu_Vilcea',146),
('Dobreta','Mehadia',75),
('Eforie','Hirsova',86),
('Fagaras','Sibiu',99),
('Hirsova','Urziceni',98),
('Iasi','Neamt',87),
('Iasi','Vaslui',92),
('Lugoj','Mehadia',70),
('Lugoj','Timisoara',111),
('Oradea','Zerind',71),
('Oradea','Sibiu',151),
('Pitesti','Rimnicu_Vilcea',97),
('Rimnicu_Vilcea','Sibiu',80),
('Urziceni','Vaslui',142)]

h=dict([('Arad',366),
('Bucharest',0),
('Craiova',160),
('Dobreta',242),
('Eforie',161),
('Fagaras',178),
('Giurgiu',77),
('Hirsova',151),
('Iasi',226),
('Lugoj',244),
('Mehadia',241),
('Neamt',234),
('Oradea',380),
('Pitesti',98),
('Rimnicu_Vilcea',193),
('Sibiu',253),
('Timisoara',329),
('Urziceni',80),
('Vaslui',199),
('Zerind',374)])

ro.add_weighted_edges_from(ro_cities)

# =============================================================================
# Search Agent
# =============================================================================

class SearchAgent(object):
    def __init__(self):
        return
    
    def BFS(self,G,start,end):
        q = [[start]]   # queue
        v = []          # visited  
        while q:
            path = q.pop(0)
            node = path[-1]
            v.append(node)
            if node==end: return path,v
            neighbours = sorted(G[node])
            for n in neighbours:
                if n not in (v+ [i[-1] for i in q]):
                    new_path = path.copy()
                    new_path.append(n)
                    q.append(new_path)
        return 'Failure'
    
    def DFS(self,G,start,end):
        q = [[start]]
        v = []
        while q:
            path = q.pop()
            node = path[-1]
            v.append(node)
            if node==end:return path,v
            neighbours = sorted(G[node],reverse=True)
            for n in neighbours:
                if n not in v:#(v + [i[-1] for i in q]):
                    #the commented out code results in suboptimal performance
                    new_path = path.copy()
                    new_path.append(n)
                    q.append(new_path)
        return 'Failure'
    
    def UCS(self,G,start,end):
        q = {(start,):0}
        v = []
        while q:
            q = dict(sorted(q.items(),key=lambda x:x[1],reverse=True))
            path,cost = q.popitem()
            node = path[-1]
            v.append(node)
            if node == end:return path,v
            neighbours = G[node]
            for n in neighbours:
                f = neighbours[n]['weight']
                if n not in v: #(v + [i[-1] for i in q])
                    #the commented out code results in suboptimal performance
                    new_path = path + (n,)
                    new_cost = f + cost
                    q[new_path]=new_cost
            #there is no need to replace existing nodes that have higher cost 
            #in queue since we are sorting by lowest cost
        return 'Failure'
    
    def GBFS(self,G,heurestics,start):
        q = {(start,):0}
        v = []
        while q:
            q = dict(sorted(q.items(),key=lambda x:x[1],reverse=True))
            path,cost = q.popitem()
            node = path[-1]
            v.append(node)
            if node=='Bucharest':return path,v
            neighbours = G[node]
            for n in neighbours:
                if n not in (v+[i[-1] for i in q]):
                    f = h[n]
                    new_path = path + (n,)
                    q[new_path]=f
        return 'Failure'    

    def A(self,G,heurestics,start):
        q = {(start,):0}
        v = []
        while q:
            q = dict(sorted(q.items(),key=lambda x:x[1],reverse=True))
            path,cost = q.popitem()
            node = path[-1]
            v.append(node)
            if node == 'Bucharest':return path,v
            neighbours = G[node]
            for n in neighbours:
                if n not in (v + [i[-1] for i in q]):
                    f = neighbours[n]['weight'] + h[n]
                    new_path = path + (n,)
                    q[new_path]=f
        return 'Failure'

def PlotPath(G,path,visited):
        plt.figure(figsize=(12,10))
        pos = nx.spring_layout(G,seed=7)
        # nodes = list(G.nodes)
        pathlist = [(path[i],path[i+1]) for i in range(len(path)-1)]
        edge_labels = dict([((i,j),k['weight']) for i,j,k in G.edges(data=True)])
        nx.draw(G,pos,with_labels=True,style='dashed',weight=0.75)
        nx.draw_networkx_nodes(G,pos,nodelist=visited,node_size=400,node_color='r')
        nx.draw_networkx_nodes(G,pos,nodelist=path,node_size=400,node_color='g')
        nx.draw_networkx_edges(G, pos, edgelist=pathlist,width=4)
        nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, label_pos = 0.5, font_size = 11)
                                       
            
# if __name__='__main__':
#     G,start,end = 

