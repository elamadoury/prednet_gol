import numpy as np
import random 
from tkinter import *
import cv2
import hickle as hkl
from PIL import Image

output_path = "D:/ShareData/gol_data/gol_random_images/all/"

random.seed(321)

mode = 'normal' # 'normal' or 'glider'
save_as_hkl = False

T = 10 #num of steps in each episode_numsode
N = 150 #num of episode_numsodes

states = list()
episode_num = 1
sources = list()

# COLS, ROWS = [20, 16]
COLS, ROWS = [160, 128]


def check(x, y):
    count = 0
    table = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    for t in table:
        xx, yy = [x + t[0], y + t[1]]
        if 0 <= xx < COLS and 0 <= yy < ROWS:
            if data[yy][xx]: count += 1
    if count == 3: return True
    if data[y][x]:
        if 2 <= count <= 3: return True
        return False
    return data[y][x]

def check_glider_move(x, y):
    if x > 0 and y > 0 and data[y-1][x-1]:
        return True
    else:
        return False

def next_turn():
    global data
    data2 = []
    for y in range(0, ROWS):
        data2.append([check(x, y) for x in range(0, COLS)])
    data = data2

def game_loop():
    global c, episode_num
    while c < T:
      states.append(np.array(data, dtype="int"))
      sources.append(str(episode_num))
      if c == T - 1:
        episode_num += 1
      next_turn()
      c += 1

def save_images(data, output_path):

  i = 0
  for array in data:
    image = Image.fromarray(array.astype('uint8'), 'RGB')
    name = output_path + str(i).zfill(5) + ".png"
    image.save(name)
    # print("saved ", name)
    i = i+1

for i in range(N):
  c = 0
  data = [] # stage data
  if mode == 'normal':
    for y in range(0, ROWS): # init stage randomly
        data.append([(random.randint(0, 9) == 0) for x in range(0, COLS)])
  if mode == 'glider':
    for y in range(0, ROWS): # init stage all zero
        data.append([(False) for x in range(0, COLS)])
    data[1][2] = True
    data[2][3] = True
    data[3][1] = True
    data[3][2] = True
    data[3][3] = True

  game_loop()

# X = np.array(list(map(lambda x: cv2.cvtColor(x.astype('uint8'), cv2.COLOR_GRAY2RGB).repeat(8, axis=0).repeat(8, axis=1)
# , states)))
X = np.array(list(map(lambda x: cv2.cvtColor(x.astype('uint8'), cv2.COLOR_GRAY2RGB), states)))
X[X == 1] = 255

if mode == 'normal':
  if save_as_hkl:
    hkl.dump(X, output_path + 'X_test.hkl')
    hkl.dump(sources, output_path + 'sources_test.hkl')
  else:
    save_images(X, output_path)

if mode == 'glider':
  if save_as_hkl:
    hkl.dump(X, output_path + 'X_test.hkl')
    hkl.dump(sources, output_path + 'sources_test.hkl')
else:
    save_images(X, output_path)




