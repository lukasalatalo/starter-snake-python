import typing
import numpy as np


class GameState:

    def __init__(self, body: list, food: list, snakes: list, height, width,
                 my_id, players, health, you):
        #print("---initializing gamestate---")
        self.height = height
        self.width = width
        self.head = body[0]
        self.body = body
        self.food = food
        #self.snakes=  snakes
        self.occupied = snakes
        self.others = []
        self.id = my_id
        self.players = players
        self.other_heads = self.other_head_positions(players)
        self.health = health
        self.you = you
        
        #self.heads = [snake[0] for snake in snakes]

    def set_occupied(self, snakes, map=None):
        for s in snakes:
            for pos in s:
                if map != None:
                    map[pos[0], pos[1]] = -1000
                self.occupied.append(pos)
        return map

    def copy_gs(self):
        body = self.body
        food = self.food
        #snakes=self.snakes
        oc = self.occupied

        gs = GameState(body, food, oc, self.width, self.height, self.id,
                       self.players, self.health, self.you)
        return gs

    def other_head_positions(self, snake):
        temp = []
        for s in snake:
            if s['id'] != self.id:
                temp.append(s['head'])
                self.others.append(s)
        return temp
