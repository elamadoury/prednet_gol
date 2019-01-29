import numpy as np
import pickle
from random import randint
from tkinter import *

mode = 'normal' # 'normal' or 'glider'

T = 10 #num of steps in each episode
N = 100 #num of episodes

COLS, ROWS = [20, 20]
CW = 20


def check(x, y):
    cnt = 0
    tbl = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    for t in tbl:
        xx, yy = [x + t[0], y + t[1]]
        if 0 <= xx < COLS and 0 <= yy < ROWS:
            if data[yy][xx]: cnt += 1
    if cnt == 3: return True
    if data[y][x]:
        if 2 <= cnt <= 3: return True
        return False
    return data[y][x]

def next_turn():
    global data
    data2 = []
    for y in range(0, ROWS):
        data2.append([check(x, y) for x in range(0, COLS)])
    data = data2

def game_loop():
    global c
    while c < T:
      s_all.append(np.array(data, dtype="int"))
      if c == T - 1:
        t_all.append(True)
      else:
        t_all.append(False)
      next_turn()
      c += 1


s_all = list()
t_all = list()

for i in range(N):
  c = 0
  data = [] # stage data
  if mode == 'normal':
    for y in range(0, ROWS): # init stage randomly
        data.append([(randint(0, 9) == 0) for x in range(0, COLS)])
  if mode == 'glinder':
    for y in range(0, ROWS): # init stage all zero
        data.append([(False) for x in range(0, COLS)])
  data[1][2] = True
  data[2][3] = True
  data[3][1] = True
  data[3][2] = True
  data[3][3] = True

  game_loop()

d = list()
d.append(s_all)
d.append(t_all)

if mode == 'normal':
  with open('lifegame_data.pickle', 'wb') as f:
      pickle.dump(d, f)
if mode == 'glinder':
  with open('lifegame_data_glider.pickle', 'wb') as f:
      pickle.dump(d, f)













