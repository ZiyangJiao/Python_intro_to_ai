# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()
    root = (problem.getStartState(), [], 0)  # node = (state, route, cost)
    stack.push(root)
    # states_record = [[problem.getStartState()]]
    states_record = []
    states_record = states_record + [problem.getStartState()]
    while not stack.isEmpty():
        (state, route, cost) = stack.pop()
        #  usage: (successor, action, stepCost) = problem.getSuccessors()
        for (successor, action, stepCost) in problem.getSuccessors(state):
            new_route = route + [action]
            new_cost = cost + stepCost
            if (not problem.isGoalState(successor)) and (successor not in states_record):
                states_record = states_record + [successor]
                new_node = (successor, new_route, new_cost)
                stack.push(new_node)
            elif problem.isGoalState(successor):
                return new_route

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    root = (problem.getStartState(), [], 0)  # node = (state, route, cost)
    queue.push(root)
    states_record = []
    states_record = states_record + [problem.getStartState()]
    while not queue.isEmpty():
        (state, route, cost) = queue.pop()
        for (successor, action, stepCost) in problem.getSuccessors(state):
            # if successor in states_record:
            #     continue
            new_route = route + [action]
            new_cost = cost + stepCost
            if (not problem.isGoalState(successor)) and (successor not in states_record):
            # if not problem.isGoalState(successor):
                states_record = states_record + [successor]
                new_node = (successor, new_route, new_cost)
                queue.push(new_node)
            elif problem.isGoalState(successor):
                return new_route

    util.raiseNotDefined()

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    pq = util.PriorityQueue()
    root = (problem.getStartState(), [], 0)  # node = (state, route, cost)
    pq.push(root, 0)  # node = (item, priority)
    # states_record = [[problem.getStartState()]]
    states_record = []
    while not pq.isEmpty():
        (state, route, cost) = pq.pop()
        if state not in states_record:
            states_record = states_record + [state]
        else:
            continue
        if problem.isGoalState(state):
            return route
        
        for (successor, action, stepCost) in problem.getSuccessors(state):
            new_route = route + [action]
            new_cost = cost + stepCost
            new_node = (successor, new_route, new_cost)
            pq.push(new_node, new_cost)
            
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    pq = util.PriorityQueue()  # priority is gCost + hCost
    root = (problem.getStartState(), [], 0)  # node = (state, route, gCost)
    states_record = []
    pq.push(root, heuristic(root[0], problem))
    while not pq.isEmpty():
        (state, route, cost) = pq.pop()
        if state not in states_record:
            states_record = states_record + [state]
        else:
            continue
        if problem.isGoalState(state):
            return route
        for (successor, action, stepCost) in problem.getSuccessors(state):
            new_route = route + [action]
            new_cost = cost + stepCost
            new_node = (successor, new_route, new_cost)
            pq.push(new_node, new_cost + heuristic(successor, problem))
            
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
