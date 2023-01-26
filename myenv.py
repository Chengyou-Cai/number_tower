import json
from tkinter import CENTER
import turtle
import random
import numpy as np
from gym import Env, spaces
from gym.envs.toy_text.utils import categorical_sample

LD = 0
RD = 1

class NumberTower(Env):

    def __init__(self,max_len=5):
        self.unit = 50
        self.max_x = max_len
        self.max_y = max_len
        self.shape = (self.max_x,self.max_y)

        self.nS = np.prod(self.shape)
        self.nA = 2

        self.box_reward = None

        self._generate_box()


        self.start_state_index = np.ravel_multi_index((0, 0), self.shape) # 0 
        print(f"start index : {self.start_state_index}, start coord : (0,0)")
        
        # Calculate transition probabilities and rewards
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s,self.shape)
            self.P[s] = {a:[] for a in range(self.nA)}
            self.P[s][LD] = self._calculate_transition_prob(current=position, delta=(1,0))
            self.P[s][RD] = self._calculate_transition_prob(current=position, delta=(1,1))

        # Calculate initial state distribution
        # We always start in state (0, 0)
        self.initial_state_distrib = np.zeros(self.nS)
        self.initial_state_distrib[self.start_state_index] = 1.0

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)
        
        self.lastaction = None ###
        self.s = None ###
        self.t = None
    
    def _generate_box(self):
        self.box_reward = np.zeros(self.shape,dtype=int)
        for i in range(self.max_x):
            for j in range(self.max_y):
                if j <= i : 
                    self.box_reward[i,j] = random.randint(1,20)
                else:
                    self.box_reward[i,j] = -100
        print(f"{type(self.box_reward)}\n",self.box_reward)

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] -1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] -1)
        coord[1] = max(coord[1], 0)
        return coord
    
    def _calculate_transition_prob(self, current, delta):
        """
        Determine the outcome for an action. Transition Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: (1.0 : prob, new_state, reward, done)
        """
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        
        r = self.box_reward[tuple(new_position)] 

        if r==-100:
            return [(1.0, self.start_state_index, -100, False)]

        terminal_state = (self.shape[0] -1, self.shape[1]-1)
        is_done = tuple(current)[0] == terminal_state[0] ### watch
        return [(1.0, new_state, r, is_done)]

    def reset(self,seed=0,return_info=False,options=None):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        if not return_info:
            return int(self.s)
        else:
            return int(self.s),{"prob":1}

    def step(self,a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, d, {"prob": p})
    
    def draw_x_line(self, y, x0, x1, color='gray'):
        assert x1 > x0
        self.t.color(color)
        self.t.setheading(0)
        self.t.up()
        self.t.goto(x0, y)
        self.t.down()
        self.t.forward(x1 - x0)

    def draw_y_line(self, x, y0, y1, color='gray'):
        assert y1 > y0
        self.t.color(color)
        self.t.setheading(90)
        self.t.up()
        self.t.goto(x, y0)
        self.t.down()
        self.t.forward(y1 - y0)

    def trans_coord(self,x,y):
        return y,(self.max_x-1)-x
    
    def draw_box(self,x,y,fillcolor='',line_color='gray'):
        x,y = self.trans_coord(x,y)

        self.t.up()
        self.t.goto(x * self.unit, y * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90)
        self.t.down()
        
        self.t.begin_fill()
        for i in range(4):
            self.t.forward(self.unit)
            self.t.right(90)
        self.t.end_fill()

    def write_text(self,x,y,text,fillcolor='',line_color='gray'):
        x,y = self.trans_coord(x,y)
        
        self.t.up()
        self.t.goto((x+0.45) * self.unit, (y+0.4) * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90)
        
        self.t.write(str(text),move=True,align=CENTER,font=("consolas",15))

    def move_player(self,state):
        if state == 0:
            self.t.up()
        else:
            self.t.down()
        
        x = state % self.max_x
        y = self.max_y - 1 - int(state / self.max_x)

        self.t.setheading(90)
        self.t.fillcolor('red')
        self.t.goto((x + 0.5) * self.unit, (y + 0.5) * self.unit)
    
    def render(self):
        if self.t == None:
            self.t = turtle.Turtle()
            
            self.wn = turtle.Screen()
            self.wn.setup(self.unit * self.max_x + 200,
                        self.unit * self.max_y + 200)
            self.wn.setworldcoordinates(0, 0, self.unit * self.max_x,
                                        self.unit * self.max_y)
            self.t.shape('circle')
            self.t.speed(0)

            self.t.width(10)
            self.t.color('gray')
            for _ in range(2):
                self.t.forward(self.max_x * self.unit)
                self.t.left(90)
                self.t.forward(self.max_y * self.unit)
                self.t.left(90)
            
            for i in range(1, self.max_y):
                self.draw_x_line(
                    y=i * self.unit, x0=0, x1=self.max_x * self.unit,color='red')
            
            for i in range(1, self.max_x):
                self.draw_y_line(
                    x=i * self.unit, y0=0, y1=self.max_y * self.unit,color='green')

            for i in range(self.max_x):
                for j in range(self.max_y):
                    if self.box_reward[i,j]<=0:
                        self.draw_box(i, j, 'black')
                    else:
                        self.write_text(i, j, text=self.box_reward[i,j])

            self.t.shape('circle')
        
        self.t.width(1)
        self.move_player(state=self.s)

def top_down_approach(shuta):
    bound = len(shuta)-1
    dp={}

    def rec(i,j):
        if dp.get(i) is None:
            dp[i] = {}
        if i == bound: # define bound
            dp[i][j] = shuta[i,j]
        else:
            dp[i][j] = max(rec(i+1,j),rec(i+1,j+1)) + shuta[i,j]
        return dp[i][j]
    
    opt = rec(0,0)
    link = []
    last_k_ = None
    for k,v in dp.items():
        if k == 0:
            k_ = max(v,key=v.get)
        else:
            t = {last_k_:v[last_k_],last_k_+1:v[last_k_+1]}
            k_ = max(t,key=t.get)
        link.append((k,k_))
        last_k_ = k_

    return opt,link

def draw_dp_link(env):
        opt,link = top_down_approach(env.box_reward)
        print(f"MaxValue:{opt}->{link}")
        
        if env.t != None:
            env.t.color("red")
            env.t.speed(3)
            for ele in link:
                s = np.ravel_multi_index(ele,env.shape)
                env.move_player(s)

if __name__ == "__main__":
    random.seed(0)
    env = NumberTower(max_len=7)
    draw_dp_link(env)


