import typing
import numpy as np

class GameState:

    def __init__(self,body:list,food:list,snakes:list,height,width):
        self.height=height
        self.width=width
        self.head=body[0]
        self.body = body
        self.food  = food
        #self.snakes=  snakes
        self.occupied = snakes
        
    def set_occupied(self,snakes,map=None):
        for s in snakes:
            for pos in s:
                if map!=None:
                    map[pos[0],pos[1]]=-1000
                self.occupied.append(pos)
        return map
    
    def copy_gs(self):
        body = self.body[:]
        food = self.food[:]
        #snakes=self.snakes
        oc= self.occupied[:]
        gs = GameState(body,food,oc,self.width,self.height)
        return gs
        


