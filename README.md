
<img src="Images/1200px-University_of_Tehran_logo.svg.png" width="100" align="left" />

<img src="Images/fanni.png" width="100" align="right"/>


<h1 style="float:center;" align="center">Computer Assignment 1</h1>
<h4 style="float:center;" align="center"><b> Navid Akbari ( 810895023 ) </b></h4>

<br>

Goal of this computer assignment is getting more familier with searching algorithms. From searching algorithms we will learn `BFS`, `IDS`, and `Astar Search`. All of these algorithms have an optimal answer but with different exectuion time. We compare these algorithm at last of this document.


```python
import time 
from collections import deque
import math
import pandas as pd
import matplotlib.pyplot as plt
from queue import PriorityQueue # It uses for Astar's frontier
import re #for finding digits in a string
```

### Problem Modeling

We need to find the steps which lead to the arrival of patients to the hospital. We need to find the optimal way of doing this. We can model the problem and change it in order to make it a search problem. Then we can use search strategies to solve the problem.

We can model our problem such that:

- **Initial State**: The map structure as read from the file

- **Actions**: Ambulance can move UP, DOWN, RIGHT, and LEFT if its way is not blocked by walls or patients

- **Transition Model**: If the ambulance goes from one place to another, it will be removed from the previous position and added to the new one in the map. If any patients exist in front of the ambulance, the patient will move further one step in the same direction. After each movement, the state will be the new map with the changed mentioned.

- **Goal State**: The goal is to have a map free of patients.

- **Path Cost**: The number of movements in order to reach the solution will be the path cost. (The depth of the solution in the search tree)

In order to keep the states and do the necessary actions on them, I have defined a class structure called State. This class has the attributes below:

- **map**: The string of map (in this class, I didn't change map data from string to other data structures like 2d array. Because it got more time to find patients and ambulance.)

- **patientsNumber**: Number of patients exists on the map in each state. This attribute is using for finding the goal state.

- **ambulancePosition**: I have used this attribute for creating a new state from the previous state.

- **patientsPosition**: This attribute is using for moving the patient if the ambulance has moved it forward.

- **hospitalsCapacities**: For calculating new capacity of hospital after patients arrive there.

The actions needed for the states and transition model are implemented as methods of the class.

- **getPossibleStates**: This method calculate possible states that are reachable from current state. **checkObstacle** and **move** are used in this method. The former one is used for checking that the new state is possible or not, and the later change the map and save it in new map.

- **isGoalState**: This method is used to find out that the state is goal state or not.

- **firstHeuristic** and **secondHeuristic**: These are huristic functions used in Astar algorithm. I explain them later.



```python
class State:
    
    def __init__(self, mapData):
        self.map = mapData
        self.ncols = self.map.find('\n')
        self.patientsNumber = self.getNumberOfPatients()
        self.patientsPosition = self.getPatientsPostions()
        self.ambulancePosition = self.getAmbulancePosition()
        self.hospitalsCapacities = self.getHospitalsCapacities()
        
    def getNumberOfPatients(self):
        return self.map.count('P')
    
    def getAmbulancePosition(self):
        return self.change_1d_to_2d(self.map.find('A'))
    
    def getPatientsPostions(self):
        return [self.change_1d_to_2d(i) for i in range(len(self.map)) if self.map.startswith('P', i)]
    
    def getHospitalsCapacities(self):
        return [int(x) for x in re.findall(r'[0-9]+', self.map)]
    
    def change_pos_to_1d(self, pos):
        return pos[1]*(self.ncols+1) + pos[0]
    
    def change_2d_to_1d(self, y, x):
        return x*(self.ncols+1) + y
    
    def change_1d_to_2d(self, i):
        x = int(i / (self.ncols + 1))
        y = i - (self.ncols + 1) * x
        return y,x
    
    def isGoalState(self):
        return True if self.patientsNumber == 0 else False
    
    def checkObstacle(self, i, j, direction):
        if i < 0 and j < 0 : 
            return False
        index = self.change_2d_to_1d(i, j)
        
        if self.map[index] == 'P':
            if direction == 'Right':
                if self.map[self.change_2d_to_1d(i+1, j)] == 'P'  or self.map[self.change_2d_to_1d(i+1, j)] == '#':
                    return False
            if direction == 'Down':
                if self.map[self.change_2d_to_1d(i, j+1)] == 'P'  or self.map[self.change_2d_to_1d(i, j+1)] == '#':
                    return False
            if direction == 'Left':
                if self.map[self.change_2d_to_1d(i-1, j)] == 'P'  or self.map[self.change_2d_to_1d(i-1, j)] == '#':
                    return False
            if direction == 'Up':
                if self.map[self.change_2d_to_1d(i, j-1)] == 'P'  or self.map[self.change_2d_to_1d(i, j-1)] == '#':
                    return False
        if self.map[index] == '#':
            return False
        
        return True
    
    def changeMap(self, map, y, x, j, i, agent):
        index_first = x*(self.ncols+1) + y
        index_sec = i*(self.ncols+1) + j
        new = map[:index_first] + ' ' + map[index_first+1:]
        return new[:index_sec] + agent + new[index_sec+1:]
    
    def move(self, i, j, direction):
        index = self.change_2d_to_1d(i, j)
        newMap = self.map
        if self.map[index] == 'P':
            x, y = i, j
            if direction == 'Right':
                tempi, tempj = i+1, j
            if direction == 'Down':
                tempi, tempj = i, j+1
            if direction == 'Left':
                tempi, tempj = i-1, j
            if direction == 'Up':
                tempi, tempj = i, j-1
            tempIndex = self.change_2d_to_1d(tempi, tempj)
            if self.map[tempIndex].isdigit():
                if self.map[tempIndex] == '1':
                    newMap = self.changeMap(newMap, x, y, tempi, tempj, ' ')
                else:
                    newMap = self.changeMap(newMap, x, y, tempi, tempj, str(int(self.map[tempIndex])-1))
            else: 
                newMap = self.changeMap(newMap, x, y, tempi, tempj, 'P')
               
        x, y = self.ambulancePosition[0], self.ambulancePosition[1]
        newMap = self.changeMap(newMap, x, y, i, j, 'A')
        return newMap
        
        
    def getPossibleStates(self): 
        states = []
        i , j = self.ambulancePosition[0], self.ambulancePosition[1]

        if self.checkObstacle(i+1, j, 'Right'):
            states.append(self.move(i+1, j, 'Right')) 
            
        if self.checkObstacle(i, j+1, 'Down'):
            states.append(self.move(i, j+1, 'Down'))
            
        if self.checkObstacle(i-1, j, 'Left'):
            states.append(self.move(i-1, j, 'Left'))
            
        if self.checkObstacle(i, j-1, 'Up'):
            states.append(self.move(i, j-1, 'Up'))
            
        return states
        
        
    def getHospitalsPosition(self):
        self.HospitalsPosition = []
        for j in self.hospitalsCapacities:
            temp = [self.change_1d_to_2d(i) for i in range(len(self.map)) if self.map.startswith(str(j), i)]
            for k in temp:
                if k not in self.HospitalsPosition:
                    self.HospitalsPosition.append(k)
        return self.HospitalsPosition
    
    
    def calculateDistance(self, x1,y1,x2,y2):  
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
        return dist  
    
    def calculateAmbulanceDistanceFromHospitals(self):
        total = 0
        for i in self.HospitalsPosition:
            total += self.calculateDistance(i[0],i[1],self.ambulancePosition[0],self.ambulancePosition[1])
        return total
    
    def calculateAmbulanceDistanceFromPatients(self):
        total = 0
        for i in self.patientsPosition:
            total += self.calculateDistance(i[0],i[1],self.ambulancePosition[0],self.ambulancePosition[1])
        return total
    
    def calculatePatientDistanceFromHospitals(self, x, y):
        result = []
        
        for i in self.HospitalsPosition:
            result.append(self.calculateDistance(x, y, i[0], i[1]))
        if len(result) == 0:
            return [0]
        return result
        
    def calculatePatientsDistanceFromHospitals(self):
        total = 0
        for i in self.patientsPosition:
            total += min(self.calculatePatientDistanceFromHospitals(i[0],i[1]))
        return total
        
    def firstHeuristic(self):
        self.getHospitalsPosition()
        return self.calculatePatientsDistanceFromHospitals() + self.calculateAmbulanceDistanceFromPatients()
    
    def secondHeuristic(self):
        self.getHospitalsPosition()
        return self.calculatePatientsDistanceFromHospitals() 
    
```

I also use a class named **Node** for having the state of the node and its level. I also keep its heuristic value for A* algorithm. I have impelement `__cmp__` and `__gt__` for comparing the heuristic values of two distinct node.


```python
class Node:
    
    def __init__(self, state, level, heuristicValue = math.inf):
        self.state = state
        self.level = level
        self.heuristicValue = heuristicValue
        
    def __cmp__(self, other):
        return cmp(self.heuristicValue, other.heuristicValue)
    
    def __gt__(self, other):
        return self.heuristicValue > other.heuristicValue
```

Now I read one of the test cases and print its information for example.


```python
mapData = open("./testcases/test2.txt", "r").read()
print(mapData)

initialState = State(mapData)
print('Ambulance Position is: ', initialState.ambulancePosition)
print('Patients position are: ', initialState.getPatientsPostions())
print('Hospitals capacities are: ', initialState.getHospitalsCapacities())
```

    ######
    #   3#
    # ## #
    #   P#
    # PP #
    #1P A#
    ######
    
    Ambulance Position is:  (4, 5)
    Patients position are:  [(4, 3), (2, 4), (3, 4), (2, 5)]
    Hospitals capacities are:  [3, 1]


## BFS


Breadth-first search(BFS) is an algorithm for traversing or searching tree or graph data structures which give us the optimal solution. It starts at the tree root and explores all of the neighbour nodes at the present depth prior to moving on to the nodes at the next depth level. As the nodes are explored based on their depth, it is guaranteed that the solution is optimal.

This algorithm has a queue as it's frontier and a set as it's explored. I have chosen the `set` data structure for the explored node as we can look up a value in a set with approximately time complexity of O(1).

The algorithm run like this:


* At first, we added initial state(map of the test cases) into the frontier and explored and then pop it from the frontier.

* Then all possible actions from the initial state which does not explore$^{*}$ before will add to the frontier

* Next, we will redo these steps until we find goal state.

$^{*}$ The given pseudo-code for BFS in the slides check that new actions were neither in explored nodes nor in the frontier. But because I use a queue for implementing frontier, searching on this data structure is not efficient. So, I used the explored set and add frontier nodes on it; therefore, I have both of the explored and frontier together.


```python
def bfs(initialState):
    allMove = 1
    node = Node(initialState,0)
    if node.state.isGoalState(): 
        return node.level, allMove, allMove
    frontier = deque([node])
    explored = set([node.state.map])
    while True : 
        if len(frontier) == 0: 
            return -1, -1, -1
        node = frontier.popleft()
        for action in node.state.getPossibleStates():
            if not action in explored:
                child = Node(State(action), node.level + 1)
                if child.state.isGoalState(): 
                    return child.level, allMove, allMove
                frontier.append(child)
                allMove += 1
                explored.add(child.state.map)
```

Now, I run one of the test cases for example.


```python
mapData = open("./testcases/test2.txt", "r").read()
initialState = State(mapData)

start = time.time() 
level, allMove, allDistinctMove = bfs(initialState)
end = time.time()

print("Time: %s seconds" % (end - start))
print("Solution's Depth: ", level)
print("Nodes Visited: ", allMove)
print("Distinct Nodes Visited: ", allDistinctMove)
```

    Time: 0.8749079704284668 seconds
    Solution's Depth:  27
    Nodes Visited:  15736
    Distinct Nodes Visited:  15736


## IDS


Iterative deepening search(IDS) is a state space/graph search strategy in which a depth-limited version of depth-first search is run repeatedly with increasing depth limits until the goal is found. IDS is optimal like breadth-first search, but uses much less memory; at each iteration, it visits the nodes in the search tree in the same order as depth-first search, but the cumulative order in which nodes are first visited is effectively breadth-first.

For running `IDS` algorithm we should saved explored node, but unlike `BFS` algorithm we should check the level(depth) of the node to make sure we should add new node to frontier or not. Because, in `IDS` algorithm it is possible to see a repeated state but in less depth. So, I make a new class named **ExploredNode** to save explored node. I have implement `__eq__` because when we want to check that if the new state is on explored set or not we should check the level of the two nodes to decide. Also, `__hash__` is used for having a class object in set data structure.




```python
class ExploredNode():
    def __init__(self, data, level):
        self.level = level
        self.data = data
        
    def __eq__(self, other):
        if self.data == other.data and self.level <= other.level:
            return True
        return False
    
    def __hash__(self):
        return hash((self.data))
```

In order to implement the `IDS` algorithm we should call limited DFS in each level, so I used two while loop for implementing this algorith. This algorithm has a stack as its frontier(for avoiding using recursion for implementation of limited DFS) and a set of **ExploredNode** as it's explored. 

The algorithm run like this:

* First while loop is using for increasing the level in each iteration. At each iteration frontier and explored data structure are reset to initial value.

* The inner loop is for implementation of limited DFS. At each round, we pop first node from stack and if it wasn't greater than our level limit we expand it and add its possible state to the frontier.


```python
def ids(initialState):
    level = 0
    allMove = 1
    node = Node(initialState, 0)
    
    if node.state.isGoalState():
        return node.level, allMove, allMove
    
    while True:
        allDistinctMove = 1
        frontier = deque([Node(initialState, 0)])
        explored = set([ExploredNode(initialState.map, 0)])
        level += 1
        
        while True:
            if len(frontier) == 0 : 
                break # Depth explored
            node = frontier.pop()
            if node.level >= level : 
                continue 
            for action in node.state.getPossibleStates():
                if not ExploredNode(action, node.level+1) in explored:
                    allMove += 1
                    if not ExploredNode(action, math.inf) in explored : 
                        allDistinctMove += 1
                    child = Node(State(action), node.level + 1)
                    if child.state.isGoalState(): 
                        return node.level+1, allMove, allDistinctMove
                    explored.add(ExploredNode(action, node.level + 1))
                    frontier.append(child)
                    
```

Now, I run one of the test cases for example.


```python
mapData = open("./testcases/test2.txt", "r").read()
initialState = State(mapData)

start = time.time()
level, allMove, allDistinctMove = ids(initialState)
end = time.time()

print("Time: %s seconds" % (end - start))

print("Solution's Depth: ", level)
print("Nodes Visited: ", allMove)
print("Distinct Nodes Visited: ", allDistinctMove)
```

    Time: 7.870615005493164 seconds
    Solution's Depth:  27
    Nodes Visited:  200659
    Distinct Nodes Visited:  6933


## A*

A* is an informed search algorithm, or a best-first search, meaning that it is formulated in terms of weighted graphs: starting from a specific starting node of a graph, it aims to find a path to the given goal node having the smallest cost. It does this by maintaining a tree of paths originating at the start node and extending those paths one edge at a time until its termination criterion is satisfied.
At each iteration of its main loop, A* needs to determine which of its paths to extend. It does so based on the cost of the path and an estimate of the cost required to extend the path all the way to the goal. Specifically, A* selects the path that minimizes:
$$
f(n) = g(n) + h(n)
$$

Where n is the next node on the path, g(n) is the cost of the path from the start node to n, and h(n) is a heuristic function that estimates the cost of the cheapest path from n to the goal. A* terminates when the path it chooses to extend is a path from start to goal or if there are no paths eligible to be extended. The heuristic function is problem-specific.

A* will give us the optimal solution if:

- The heuristic function is admissible - for tree search

- The heuristic function is consistent - for graph search


### First Heuristic

For the first heuristic, I calculate the manhattan distance between patients and nearest hospital from each patient and add it to the ambulance manhattan distance from each patients. 
`self.calculatePatientsDistanceFromHospitals() + self.calculateAmbulanceDistanceFromPatients()`

From definition a heuristic is admissible if: $h(n) \leq g(n) $ which $h(n)$ is heuristic function that estimates the cost of the cheapest path from n to the goal and $g(n)$ is actual path cost to goal.  

My proposed heuristic is admissible because:

**PROVE**: In my heuristic, I calculate the minimum manhattan path cost for taking each patient to the hospital. But in reality,it is probable that there exist an obstacles in the way or the agents may choose to go in opposite directions, etc. Also, I calculate the manhattan distance and the real path is greater.

### Second Heuristic

For the second heuristic, I calculate the manhattan distance between patients and nearest hospital from each patient. 
`self.calculatePatientsDistanceFromHospitals()`

My proposed heuristic is admissible because:

**PROVE**: Previous heuristic was admissble, therefore this heuristic is admissble too. Because its less than previous one.

### Heuristic Comparison

As the table in the end of the document shows, the first heuristic is much more efficient because its calculate $h(n)$ better. And its more near to the actual cost.

### Algorithm

This algorithm is very similar to `BFS` algorithm, but it uses a priority queue as it's frontier instead. Also, I used set of **ExploredNode** as it's explored data structure due to the same reason in IDS algorithm. I add one attribute to the Node class for keep tracking of the heuristic value in each node.

The algorithm run like this:
 
* At first we put intial state into frontier and then get$^{*}$ it from the priority queue. 
* Then all possible actions from initial values which does not explored before, will add to the frontier 
* and next we will redo these steps until we find goal state.

$^{*}$ The get method give us the node with minimum heuristic value, because I overwrite the `__gt__` method in Node class.


```python
def aStar(initialState, heuristicFunction):
    allMove = 1
    allDistinctMove = 1
    frontier = PriorityQueue()
    if heuristicFunction == 'first':
        node = Node(initialState, 0, initialState.firstHeuristic() + 0)
    else:
        node = Node(initialState, 0, initialState.secondHeuristic() + 0)
    frontier.put(node)
    if node.state.isGoalState(): 
        return node.level, allMove, allDistinctMove
    explored = set([ExploredNode(node.state.map, 0)])
    while True : 
        if frontier.empty(): 
            return -1, -1, -1
        node = frontier.get()
        for action in node.state.getPossibleStates():
            if not ExploredNode(action, node.level + 1) in explored :
                allMove += 1
                if not ExploredNode(action, math.inf) in explored :
                    allDistinctMove += 1
                childState = State(action)
                if childState.isGoalState(): 
                    return node.level + 1, allMove, allDistinctMove
                
                if heuristicFunction == 'first':
                    child = Node(childState, node.level + 1, node.level + 1 + childState.firstHeuristic())
                else:
                    child = Node(childState, node.level + 1, node.level + 1 + childState.secondHeuristic())
                
                frontier.put(child)
                explored.add(ExploredNode(child.state.map, child.level))
```

Now, I run one of the test cases for first heuristic for example.


```python
mapData = open("./testcases/test2.txt", "r").read()
initialState = State(mapData)

start = time.time()
level, allMove, allDistinctMove = aStar(initialState, 'first')
end = time.time()

print("Time: %s seconds" % (end - start))

print("Solution's Depth: ", level)
print("Nodes Visited: ", allMove)
print("Distinct Nodes Visited: ", allDistinctMove)
```

    Time: 0.6154098510742188 seconds
    Solution's Depth:  27
    Nodes Visited:  5932
    Distinct Nodes Visited:  5477


Now, I run one of the test cases for second heuristic for example.


```python
mapData = open("./testcases/test2.txt", "r").read()
initialState = State(mapData)

start = time.time()
level, allMove, allDistinctMove = aStar(initialState, 'second')
end = time.time()

print("Time: %s seconds" % (end - start))

print("Solution's Depth: ", level)
print("Nodes Visited: ", allMove)
print("Distinct Nodes Visited: ", allDistinctMove)
```

    Time: 0.8996541500091553 seconds
    Solution's Depth:  27
    Nodes Visited:  12626
    Distinct Nodes Visited:  11821


## Comparison

### BFS

- Complete: Yes (If branching factor is finite)

- Optimal: Yes (If cost of the edges are equal)

- Time: O($b^d$) where b is the branching factor and d is the solution's depth.

- Space: O($b^d$)


### IDS

- Complete: Yes (If branching factor is finite)

- Optimal: Yes (If cost of the edges are equal.)

- Time: ($d+1$)$b^0$ + ($d$)$b^1$ + ($d-1$)$b^2$ + ... + $b^d$ = O($b^d$)

- Space: O($bd$)


### A*

- Complete: Yes 

- Optimal: Yes 

- Time: Number of nodes for which f(n) ≤ C* where C* is the optimal path cost (exponential). It actually depends on the heuristic and it reduces when the heuristic gets closer to the actual cost. 

- Space: exponential 

To sum up we can say here A* and BFS can be good answers for this problem. They have near time complexity and have less seen state.


```python
def findAverageForAlgorithm(alg, initialState, heuristicFunction=''):
    meanTime, allMove, allDistinctMove, level = 0, 0, 0, 0;
    for i in range(0, 3):
        if alg == aStar:
            start = time.time()
            level, allMove, allDistinctMove = alg(initialState, heuristicFunction)
            end = time.time()
        else:
            start = time.time()
            level, allMove, allDistinctMove = alg(initialState)
            end = time.time()
        
        meanTime += end - start
    return level, allMove, allDistinctMove, meanTime/3

tables = []

for path in ['./testcases/test1.txt', './testcases/test2.txt', './testcases/test3.txt']:
    table = pd.DataFrame([], index=['BFS','IDS', 'First A*', 'Second A*'],
                     columns=['Answer Cost', 'Total States', 'Total Distinct States', 'Execution Time'])
    mapData = open(path, "r").read()
    initialState = State(mapData)
    
    level, allMove, allDistinctMove, meanTime = findAverageForAlgorithm(bfs, initialState)
    table.loc['BFS']['Answer Cost'] = level
    table.loc['BFS']['Total States'] = allMove
    table.loc['BFS']['Total Distinct States'] = allDistinctMove 
    table.loc['BFS']['Execution Time'] = meanTime
   
    level, allMove, allDistinctMove, meanTime = findAverageForAlgorithm(ids, initialState)
    table.loc['IDS']['Answer Cost'] = level
    table.loc['IDS']['Total States'] = allMove
    table.loc['IDS']['Total Distinct States'] = allDistinctMove 
    table.loc['IDS']['Execution Time'] = meanTime
    
    level, allMove, allDistinctMove, meanTime = findAverageForAlgorithm(aStar, initialState, 'first')
    table.loc['First A*']['Answer Cost'] = level
    table.loc['First A*']['Total States'] = allMove
    table.loc['First A*']['Total Distinct States'] = allDistinctMove 
    table.loc['First A*']['Execution Time'] = meanTime
    
    level, allMove, allDistinctMove, meanTime = findAverageForAlgorithm(aStar, initialState, 'second')
    table.loc['Second A*']['Answer Cost'] = level
    table.loc['Second A*']['Total States'] = allMove
    table.loc['Second A*']['Total Distinct States'] = allDistinctMove 
    table.loc['Second A*']['Execution Time'] = meanTime
    
    tables.append(table)
    
```

### Test case 1


```python
tables[0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Answer Cost</th>
      <th>Total States</th>
      <th>Total Distinct States</th>
      <th>Execution Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BFS</th>
      <td>11</td>
      <td>510</td>
      <td>510</td>
      <td>0.016871</td>
    </tr>
    <tr>
      <th>IDS</th>
      <td>11</td>
      <td>1993</td>
      <td>497</td>
      <td>0.0606446</td>
    </tr>
    <tr>
      <th>First A*</th>
      <td>11</td>
      <td>255</td>
      <td>251</td>
      <td>0.0135767</td>
    </tr>
    <tr>
      <th>Second A*</th>
      <td>11</td>
      <td>553</td>
      <td>549</td>
      <td>0.0300509</td>
    </tr>
  </tbody>
</table>
</div>



### Test case 2


```python
tables[1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Answer Cost</th>
      <th>Total States</th>
      <th>Total Distinct States</th>
      <th>Execution Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BFS</th>
      <td>27</td>
      <td>15736</td>
      <td>15736</td>
      <td>0.672707</td>
    </tr>
    <tr>
      <th>IDS</th>
      <td>27</td>
      <td>200659</td>
      <td>6933</td>
      <td>6.97383</td>
    </tr>
    <tr>
      <th>First A*</th>
      <td>27</td>
      <td>5932</td>
      <td>5477</td>
      <td>0.389792</td>
    </tr>
    <tr>
      <th>Second A*</th>
      <td>27</td>
      <td>12626</td>
      <td>11821</td>
      <td>0.767281</td>
    </tr>
  </tbody>
</table>
</div>



### Test case 3


```python
tables[2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Answer Cost</th>
      <th>Total States</th>
      <th>Total Distinct States</th>
      <th>Execution Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BFS</th>
      <td>39</td>
      <td>64048</td>
      <td>64048</td>
      <td>2.3287</td>
    </tr>
    <tr>
      <th>IDS</th>
      <td>39</td>
      <td>1240435</td>
      <td>15978</td>
      <td>53.0919</td>
    </tr>
    <tr>
      <th>First A*</th>
      <td>39</td>
      <td>22691</td>
      <td>21198</td>
      <td>1.59182</td>
    </tr>
    <tr>
      <th>Second A*</th>
      <td>39</td>
      <td>60791</td>
      <td>58546</td>
      <td>4.05313</td>
    </tr>
  </tbody>
</table>
</div>


