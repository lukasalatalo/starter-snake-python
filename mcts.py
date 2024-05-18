import numpy as np
import random
from typing import List, Dict, Tuple
from gamestate import GameState
import math
import time


def matrix_to_board(matrix_pos, height):
    row, col = matrix_pos

    return (col, height - 1 - row)


def board_to_matrix(board_pos, height):
    x, y = board_pos

    return (height - 1 - y, x)


class Node:

    def __init__(self, body: list):
        self.children: List[Node] = []
        self.parent: Node = None
        self.action: str = ""
        self.value: float = 0
        self.reward: float = 0
        self.num_visits: int = 0
        self.location: list = body[0]
        self.body = body


class MCTS:

    def __init__(self, game_state: GameState, timeout):
        self.game_state = game_state
        self.length = self.game_state.you["length"]
        self.timeout = timeout
        self.blank_home_score: int = 1
        self.blank_enemy_score: int = 5
        self.food_score: int = self.calculate_food_value()
        self.death_penalty: int = -999999999
        self.episode: int = 1
        self.depth: int = 10
        self.tree_depth: int = 0
        self.tree_depth_thres: int = 10
        self.epsilon: float = 0.01
        self.gamma: float = 0.9
        self.root: Node = Node(game_state.body)
        self.location = self.root.location
        self.closed: list = [self.root.location]
        self.reward_map = np.zeros(
            (self.game_state.width, self.game_state.height))
        self.build_rewards()

    def calculate_food_value(self):
        # Calculate the dynamic food value based on health
        return 10000 - (9900 / 99) * (self.game_state.health - 1)

    def build_rewards(self):
        """for o in self.game_state.occupied:
            indx = (o[0] + 1) * (o[1] + 1)
            indices = self.find_radius_ind(self.game_state.width,
                                           self.game_state.height, indx, 3)
            for ind in indices:
                if self.is_in_bounds(
                        self.game_state.width, self.game_state.height, ind[0],
                        ind[1]) and self.reward_map[ind[0], ind[1]] > -1000 * (
                            abs(np.linalg.norm(indx - ind)) / 3)**2:
                    self.reward_map[ind[0], ind[1]] = -1000 * (
                        abs(np.linalg.norm(indx - ind)) / 3)**2"""
        print("-----building rewards-----")
        for player in self.game_state.players:

            for o in player['body']:
                row, col = board_to_matrix([o['x'], o['y']],
                                           self.game_state.height)
                print(f"x: {o['x']}, y: {o['y']}")
                print(f"row: {row}, col: {col}")
                self.reward_map[row][col] = -10000

        # Set a high penalty for potential head-to-head collision positions
        print("-----otherheads-----")
        for i, enemy in enumerate(self.game_state.other_heads):
            print(f"enemy: {enemy} for {i}")
            head = enemy  # Assuming the first element is the head of the enemy snake
            potential_positions = [
                (head['x'] + 1, head['y']),  # Move right
                (head['x'] - 1, head['y']),  # Move left
                (head['x'], head['y'] + 1),  # Move up
                (head['x'], head['y'] - 1)  # Move down
            ]

            bigger = 0.001 if self.length > self.game_state.others[i][
                'length'] else -1
            print("First checkpoint")
            print(f"bigger: {1 * bigger}")
            for pos in potential_positions:
                if self.is_in_bounds(self.game_state.width,
                                     self.game_state.height, pos[1], pos[0]):
                    x, y = board_to_matrix(pos, self.game_state.height)
                    self.reward_map[x][y] = 10000 * bigger

        
        food_value = self.calculate_food_value() * 0.01
        for f in self.game_state.food:
            x, y = f
            row, col = board_to_matrix((x, y), self.game_state.height)
            if self.is_in_bounds(self.game_state.width, self.game_state.height, col, row):
                self.reward_map[row][col] += food_value
                #print(f"Food: Board ({x}, {y}) -> Matrix ({row}, {col})")



        
        self.print_reward_map1()

    def print_reward_map(self):
        print("Reward Map:")
        for row in self.reward_map:
            print(" ".join(f"{value:6.2f}" for value in row))

    def print_reward_map1(self):
        print("Reward Map:")
        for row in self.reward_map:
            row_str = "|"
            for value in row:
                if value < 0:
                    row_str += " X |"
                elif value > 0:
                    row_str += " O |"
                else:
                    row_str += "   |"
            print(row_str)

    def is_in_bounds(self, width, height, x, y):
        """Returns weather (x, y) is inside grid_map or not."""
        if x >= 0 and x < width:
            if y >= 0 and y < height:
                return True
        return False

    def find_radius_ind(self, width, height, center, radius):
        indices = np.indices([width, height]).reshape(2, -1).T
        return indices[np.square(indices -
                                 np.array(center)).sum(1) <= radius**2]

    def start_tick_timer(self):
        self.tick_start_time = time.time()

    def MCTSSearch(self):
        self.start_tick_timer()
        episode = 0
        while episode < self.episode and (
                time.time() - self.tick_start_time) * 1000 < self.timeout:
            leaf = self.Traverse(self.root)
            self.Rollout(leaf)
            self.Backprop(leaf)
            episode += 1
        bestAction = self.ChooseBest(self.root)
        return bestAction

    def ChooseBest(self, root):
        bestAction = None
        maxQ = float("-inf")
        for child in root.children:
            if child.value >= maxQ:
                maxQ = child.value
                bestAction = child.action
        return bestAction

    def Backprop(self, node):
        while node is not None:
            maxQ = float("-inf")
            for child in node.children:
                if child.value > maxQ:
                    maxQ = child.value
            value = node.value
            node.value = node.reward + self.gamma * maxQ
            node.value = (value * node.num_visits +
                          node.value) / (node.num_visits + 1)
            node.num_visits += 1
            if node.location == self.location:
                break
            node = node.parent

    def Expand(self, pos, body, action, parent):
        body = body.copy()
        if body == None:
            return
        body.insert(0, pos)

        node = Node(body)
        node.action = action

        node.reward = self.GetReward(self.game_state, pos)
        if node.reward != self.food_score:
            node.body.pop(-1)
        parent.children.append(node)
        node.parent = parent
        self.closed.append(node.location)
        return node

    def Simulate(self, node):

        curPos = node.location
        t = 0
        discountedReward = 0
        closed = list(self.closed)
        gameState = self.game_state
        while t < self.depth:
            closed.append(curPos)
            r_t = self.GetReward(gameState, curPos)
            discountedReward += math.pow(self.gamma, t) * r_t
            body = node.body
            body.insert(0, curPos)
            gameState = gameState.copy_gs()

            if r_t != self.food_score:
                body.pop(-1)
            else:
                if curPos in gameState.food:
                    gameState.food.remove(curPos)
                    print(f"Food removed at {curPos}")
                else:
                    print(f"Attempted to remove food at {curPos}, but it was not found in the list")
                
            gameState.body = body
            node = Node(body)
            neighbours = self.GetNeighbours(gameState, node)

            candPos = list(neighbours.values())
            candPos = [pos for pos in candPos if pos not in closed]
            if len(candPos) == 0:
                break
            curPos = random.choice(candPos)
            t += 1
        node.num_visits += 1
        return discountedReward

    def Rollout(self, leaf):

        neighbours = self.GetNeighbours(self.game_state, leaf)
        print(neighbours)
        expandedNodes = []
        for action, pos in neighbours.items():
            if list(pos) not in self.closed:
                expandedNodes.append(
                    self.Expand(list(pos), leaf.body, str(action), leaf))
        if not expandedNodes:
            expandedNodes.append(leaf)
        nodeSim = random.choice(expandedNodes)
        reward = self.Simulate(nodeSim)
        nodeSim.value = nodeSim.reward + self.gamma * reward

    def EpsilonGreedyPolicy(self, parent):
        children = parent.children
        maxQ = float("-inf")
        bestChild = None
        for child in children:
            if child.value >= maxQ:
                maxQ = child.value
                bestChild = child
        if len(children) == 1:
            return bestChild
        childrenCopy = list(children)
        childrenCopy.remove(bestChild)
        if random.random() < self.epsilon:
            return random.choice(children)
        return bestChild

    def Traverse(self, root):
        if not root.children or self.tree_depth >= self.tree_depth_thres:
            return root
        child = self.EpsilonGreedyPolicy(root)
        self.tree_depth += 1
        return self.Traverse(child)

    def GetReward(self, gameState, pos):
        reward = 0
        '''if pos in gameState.food:
            reward += self.food_score
        else:
            reward += self.reward_map[pos[0], pos[1]]'''
        reward += self.reward_map[pos[0], pos[1]]
        return reward

    def GetNeighbours(self, gs, n):
        neighbours = dict()
        pos = n.location
        up = [pos[0], pos[1] + 1]
        down = [pos[0], pos[1] - 1]
        right = [pos[0] + 1, pos[1]]
        left = [pos[0] - 1, pos[1]]

        if up[1] < gs.height and up not in gs.occupied and up not in gs.body:
            neighbours["up"] = up
        if down[1] >= 0 and down not in gs.occupied and down not in gs.body:
            neighbours["down"] = down
        if right[
                0] < gs.width and right not in gs.occupied and right not in gs.body:
            neighbours["right"] = right
        if left[0] >= 0 and left not in gs.occupied and left not in gs.body:
            neighbours["left"] = left

        return neighbours
