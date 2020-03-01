# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
    
    def getAction(self, gameState):
        """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        
        "Add more of your code here if you want to"
        
        return legalMoves[chosenIndex]
    
    def evaluationFunction(self, currentGameState, action):
        """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        "*** YOUR CODE HERE ***"
        # consider food location and ghosts
        # cost: move: -1; eating food: +100; eating ghost: +200; approaching ghost: -1000
        value = 0
        # for food
        nextFoodStates = newFood.asList()
        nearestFoodDistance = 0
        if not len(nextFoodStates) == 0:
            nearestFoodDistance = min([util.manhattanDistance(nextFood, newPos) for nextFood in nextFoodStates])
            # each dot will be deducted 100 points
            value = value - nearestFoodDistance - (len(nextFoodStates) - 1) * 100
        # for ghost
        # if there is remaining scared time
        if min(newScaredTimes) > 0:
            value += 500
        # if approaching ghosts
        nearestGhostDistance = min([util.manhattanDistance(nextGhost.getPosition(), newPos) for nextGhost in newGhostStates])
        if nearestGhostDistance <= 1 and min(newScaredTimes) == 0:
            value -= 1000
        # if pacMan goes around repeatly
        curDirection = currentGameState.getPacmanState().getDirection()
        nextDirection = successorGameState.getPacmanState().getDirection()
        directionTable = {
            'East': 'West',
            'West': 'East',
            'North': 'South',
            'South': 'North',
            'Stop': None
        }
        if directionTable[curDirection] == nextDirection and nearestGhostDistance > 1:
            value -= 100
        if action == 'Stop':
            value -= 100
        
        value += successorGameState.getScore() - currentGameState.getScore()
        
        return value
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """
    
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
  """
    
    def getAction(self, gameState):
        """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
        "*** YOUR CODE HERE ***"
        
        def gameIsDone(gameState):
            if gameState.isWin() or gameState.isLose():
                return True
            else:
                return False
        
        def getActionPacman(agentIndex, depth, gameState):
            if gameIsDone(gameState) or depth == self.depth:
                return self.evaluationFunction(gameState), 'None'
            nextActions = gameState.getLegalActions(agentIndex)
            if len(nextActions) == 0:
                return self.evaluationFunction(gameState), 'None'
            
            maxScore = -1000000
            maxAction = 'None'
            # compare each action to get maxScore
            for action in nextActions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                nextScore = getActionGhost(agentIndex + 1, depth, nextState)
                if maxScore < nextScore[0]:
                    maxScore = nextScore[0]
                    maxAction = action
            return maxScore, maxAction
        
        def getActionGhost(agentIndex, depth, gameState):
            if gameIsDone(gameState) or depth == self.depth:
                return self.evaluationFunction(gameState), 'None'
            nextActions = gameState.getLegalActions(agentIndex)
            if len(nextActions) == 0:
                return self.evaluationFunction(gameState), 'None'
            
            minScore = 1000000
            minAction = 'None'
            # compare each action to get minScore
            for action in nextActions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                # if current ghost is the last one need to be considered, then the next round should be pacMan; otherwise, we consider the remaining ghosts
                if agentIndex == gameState.getNumAgents() - 1:
                    nextScore = getActionPacman(0, depth + 1, nextState)
                else:
                    nextScore = getActionGhost(agentIndex + 1, depth, nextState)
                if minScore > nextScore[0]:
                    minScore = nextScore[0]
                    minAction = action
            return minScore, minAction
        
        # main function
        return getActionPacman(0, 0, gameState)[1]
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
  """
    
    def getAction(self, gameState):
        """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
        "*** YOUR CODE HERE ***"
        
        def gameIsDone(gameState):
            if gameState.isWin() or gameState.isLose():
                return True
            else:
                return False
        
        def getActionPacman(agentIndex, depth, gameState, alpha, beta):
            if gameIsDone(gameState) or depth == self.depth:
                return self.evaluationFunction(gameState), 'None'
            nextActions = gameState.getLegalActions(agentIndex)
            if len(nextActions) == 0:
                return self.evaluationFunction(gameState), 'None'
            
            maxScore = -1000000
            maxAction = 'None'
            # compare each action to get maxScore
            for action in nextActions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                nextScore = getActionGhost(agentIndex + 1, depth, nextState, alpha, beta)
                if nextScore[0] > beta:
                    return nextScore[0], action
                if maxScore < nextScore[0]:
                    maxScore = nextScore[0]
                    maxAction = action
                alpha = max(alpha, nextScore[0])
            return maxScore, maxAction
        
        def getActionGhost(agentIndex, depth, gameState, alpha, beta):
            if gameIsDone(gameState) or depth == self.depth:
                return self.evaluationFunction(gameState), 'None'
            nextActions = gameState.getLegalActions(agentIndex)
            if len(nextActions) == 0:
                return self.evaluationFunction(gameState), 'None'
            
            minScore = 1000000
            minAction = 'None'
            # compare each action to get minScore
            for action in nextActions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                # if current ghost is the last one need to be considered, then the next round should be pacMan; otherwise, we consider the remaining ghosts
                if agentIndex == gameState.getNumAgents() - 1:
                    nextScore = getActionPacman(0, depth + 1, nextState, alpha, beta)
                else:
                    nextScore = getActionGhost(agentIndex + 1, depth, nextState, alpha, beta)
                if nextScore[0] < alpha:
                    return nextScore[0], action
                if minScore > nextScore[0]:
                    minScore = nextScore[0]
                    minAction = action
                beta = min(beta, nextScore[0])
            return minScore, minAction
        
        # main function
        return getActionPacman(0, 0, gameState, -1000000, 1000000)[1]
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
  """
    
    def getAction(self, gameState):
        """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
        "*** YOUR CODE HERE ***"
        
        def gameIsDone(gameState):
            if gameState.isWin() or gameState.isLose():
                return True
            else:
                return False
        
        def getActionPacman(agentIndex, depth, gameState):
            if gameIsDone(gameState) or depth == self.depth:
                return self.evaluationFunction(gameState), 'None'
            nextActions = gameState.getLegalActions(agentIndex)
            if len(nextActions) == 0:
                return self.evaluationFunction(gameState), 'None'
            
            maxScore = -1000000
            maxAction = 'None'
            # compare each action to get maxScore
            for action in nextActions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                nextScore = getActionGhost(agentIndex + 1, depth, nextState)
                if maxScore < nextScore[0]:
                    maxScore = nextScore[0]
                    maxAction = action
            return maxScore, maxAction
        
        def getActionGhost(agentIndex, depth, gameState):
            if gameIsDone(gameState) or depth == self.depth:
                return self.evaluationFunction(gameState), 'None'
            nextActions = gameState.getLegalActions(agentIndex)
            if len(nextActions) == 0:
                return self.evaluationFunction(gameState), 'None'
            
            expectiScore = 0
            expectiAction = 'None'
            uniformProb = 1.0 / len(nextActions)
            # compare each action to get minScore
            for action in nextActions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                # if current ghost is the last one need to be considered, then the next round should be pacMan; otherwise, we consider the remaining ghosts
                if agentIndex == gameState.getNumAgents() - 1:
                    nextScore = getActionPacman(0, depth + 1, nextState)
                else:
                    nextScore = getActionGhost(agentIndex + 1, depth, nextState)
                expectiScore += uniformProb * nextScore[0]
                expectiAction = action
            return expectiScore, expectiAction
        
        # main function
        return getActionPacman(0, 0, gameState)[1]
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Firstly, we count the number of remaining food and compute the total distance between pacMan and these food. The larger total distance and the more remaining food indicate the smaller score
    Secondly, we compute the total distance between pacMan and ghosts
    Thirdly, we find the status of panMan. If pacMan is aggressive (sum of scared time > 0), the farther distance between pacMan and ghosts, the lower score it should be; otherwise, the farther distance indicates the higher score
  """
    "*** YOUR CODE HERE ***"
    if currentGameState.isLose():
        return -10000
    if currentGameState.isWin():
        return 10000
    curPos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    # food
    foodLeft = currentGameState.getFood().asList()
    if len(foodLeft) != 0:
        foodDistance = sum([util.manhattanDistance(food, curPos) for food in foodLeft])
    if foodDistance != 0:
        score += 1.0 / foodDistance + len(foodLeft)
    # ghost
    ghosts = currentGameState.getGhostStates()
    ghostDistance = sum([util.manhattanDistance(ghost.getPosition(), curPos) for ghost in ghosts])
    # status
    capsuleLeft = len(currentGameState.getCapsules())
    scaredLeft = sum([ghost.scaredTimer for ghost in ghosts])
    if scaredLeft > 0:
        score += scaredLeft - ghostDistance
    else:
        score += ghostDistance + capsuleLeft
    return score
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest
  """
    
    def getAction(self, gameState):
        """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
