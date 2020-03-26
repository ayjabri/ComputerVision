from itertools import product
from queue import deque
import sys
import copy
import time
from tqdm import tqdm

class Suduko(object):
    def __init__(self,data):
        
        # A set of variables, X = {X1, X2, ··· Xn}
        self.variables = list(product(range(9),range(9)))
        self.domains = {}
        self.neighbours = {}
        self.arcs = {}
        
        # Define the set of domains for each variable: D = {D1, D2, ··· Dn}
        for idx,Xi in enumerate(self.variables):
            self.domains[Xi] = list(range(1,10)) if int(data[idx]) == 0 else [int(data[idx])]
        
        # Neighbouring cells
        for x,y in self.variables:
            self.neighbours[x,y] = ([(i,y) for i in range(9) if i!=x] + 
                    [(x,j) for j in range(9) if j!=y] + 
                    [(x//3*3+i,y//3*3+j) for i in range(3) for j in range(3) if (x//3*3+i!=x and y//3*3+j!=y)])
        
        # Define the Arcs: pairs of cells and their neighbours that will be used to start the queue
        for Xi in self.neighbours:
            self.arcs[Xi] = [(Xi,Xj) for Xj in self.neighbours[Xi]]
        return
        
    def isSolved(self):
        if all(len(d)==1 for d in self.domains.values()):
            return True
        return False
# Found out that copy.deepcopy() works better     
    # def cloneDomains(self):
    #     clone = {}
    #     for key in self.domains:
    #         kValues = []
    #         for v in self.domains[key]:
    #             kValues.append(v)
    #         clone[key] = kValues
    #     return clone
    
    def getNeighbours(self,Xi):
        return self.neighbours[Xi]
    
    def printPuzzle(self):
        x_keys = y_keys = list(range(9))
        result = "  0 1 2 3 4 5 6 7 8 \n"
        result += " +-----+-----+-----+\n"
        for y in range(9):
            result += str(y_keys[y]) + "|"
            for x in range(9):
                domain = self.domains[y_keys[y] , x_keys[x]]
                if (len(domain) == 1):
                    val = domain[0]
                else:
                    val = 0
                result += str(val)
                if (x == 2 or x == 5 or x == 8):
                    result += "|"
                else:
                    result += " "
            if (y == 2 or y == 5 or y == 8):
                result += "\n +-----+-----+-----+\n"
            else:
                result += "\n"
        return print(result)
       
    def getAgenda(self):
        result = set()
        for x in self.neighbours.keys():
            for y in self.neighbours[x]:
                result.add((x,y))
        return sorted(result)

# =============================================================================
# AC 3 Algorithm
# =============================================================================
        
class AC3(object):

    def __init__(self, csp):
        self.csp = csp
        return

    #A set of constraints C that specify allowable combinations of values
    def revise(self, Xi, Xj):
        revised = False
        for x in self.csp.domains[Xi]:
            if not any(x!=t for t in self.csp.domains[Xj]):
                self.csp.domains[Xi].remove(x)
                revised = True
        return revised
  
    # returns false if an inconsistency is found, otherwise true
    def solve(self):
        # queue - a queue of arcs, initially all the CSP arcs
        agenda = deque(self.csp.getAgenda())
        while agenda:
            (Xi, Xj) = agenda.popleft()
            if (self.revise(Xi, Xj)):
                if (len(self.csp.domains[Xi]) == 0):
                    return False
                for Xk in self.csp.neighbours[Xi]:
                    agenda.append((Xk, Xi))
        return True

# =============================================================================
# BackTrack Solver
# =============================================================================

class BTS(object):
    def __init__(self,csp):
        self.csp = csp
        self.unassigned = {}
        for Xi in self.csp.domains:
            self.unassigned[Xi] = len(self.csp.domains[Xi])>1
        return

    def getUnassigned(self):
        minKey = None
        minValues = None
        
        for key in self.unassigned.keys():
            if self.unassigned[key] == True:
                values = self.csp.domains[key]
                if minValues == None or len(values) < len(minValues):
                    minKey = key
                    minValues = values
        return (minKey, minValues)

   # return true when for every value x of X, there is some allowed y
    def isConsistent(self, key, value):
        for Xn in self.csp.neighbours[key]:
            values = self.csp.domains[Xn]
            if (len(values) == 1 and values[0] == value):
                return False
        return True
    
    def search(self, depth):
        # check if the terminal state is true
        if (self.csp.isSolved()):return True
        # Pick the next unassigned cell witht he least number of values
        (key, values) = self.getUnassigned()

        # for each value in order-domain-values(csp, assigment, csp)
        for value in values:
            if (self.isConsistent(key, value)):
#                 print(key,value)
                savedValues = copy.deepcopy(self.csp.domains)
                self.csp.domains[key] = [value]
                self.unassigned[key] = False
                ac3 = AC3(self.csp)
                ac3.solve()
                if self.search(depth + 1): 
                    return True
                # Backtrack if search didn't solve
                self.unassigned[key] = True
                self.csp.domains = savedValues
        
        return False
    def solve(self):
        return self.search(1)
    
def solve_n_print(puzzle):
    csp = Suduko(puzzle.strip())
    ac3 = AC3(csp)
    bts = BTS(csp)
    ac3.solve()
    if csp.isSolved():
        method = 'AC3'
    else:
        bts.solve()
        if csp.isSolved():
            method = 'BTS'
        else:
            method = 'Error..................'
    return csp,method

def main():
    start_time = time.process_time()
    if sys.argv[1] != 'all':    
        puzzle = sys.argv[1]
        csp,method = solve_n_print(puzzle)
        print(f'Method: {method}')
        csp.printPuzzle()
    else:
        with open('sudokus_start.txt','r') as p:
            puzzles = p.readlines()
    
        for idx,puzzle in tqdm(enumerate(puzzles)):
            csp,method = solve_n_print(puzzle)
            print(f'{idx}: {method}')
            csp.printPuzzle()
            with open('output.txt','a') as file:
                file.write(''.join([str(elem[0]) for elem in csp.domains.values()]))
                file.write(' BTS\n')
    print(time.process_time()-start_time)
     
if __name__ =='__main__':
    main()