"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example. The RL is in RL_brain.py.
learning from: morvanzhou
"""
import sys
import time

import numpy as np

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40  # pixels 像素
MAZE_H = 4  # grid height  格线
MAZE_W = 4  # grid weight


class Maze(tk.Tk, object):

    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']  # 上 下 左 右 这里顺序有关系
        self.n_actions = len(self.action_space)
        self.title = 'maze'
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))  # 指定主框体大小
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # create grids 以窗口左上角为起始点，坐标为（0,0）
        for c in range(0, MAZE_W*UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H*UNIT
            # 画直线 -- 横线
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H*UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W*UNIT, r
            # 画直线 -- 竖线
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin_arr = np.array([20, 20])

        # create hell 绘制矩形((a,b,c,d),值为左上角和右下角的坐标)
        hell_center = origin_arr + np.array([UNIT*2, UNIT])
        self.hell = self.canvas.create_rectangle(
            hell_center[0] - 15, hell_center[1] - 15,
            hell_center[0] + 15, hell_center[1] + 15,
            fill='black'
        )

        hell2_center = origin_arr + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black'
        )

        # create oval 创建椭圆
        oval_center = origin_arr + UNIT*2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow'
        )

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin_arr[0] - 15, origin_arr[1] - 15,
            origin_arr[0] + 15, origin_arr[1] + 15,
            fill='red'
        )

        self.canvas.pack()

    def reset(self):
        # 刷新即可看到图像的移动，为了使多次移动变得可视，最好加上time.sleep()函数；
        self.update()
        time.sleep(0.5)

        # 删除绘制的图形；
        self.canvas.delete(self.rect)

        origin_arr = np.array([20, 20])
        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin_arr[0] - 15, origin_arr[1] - 15,
            origin_arr[0] + 15, origin_arr[1] + 15,
            fill='red'
        )
        # 返回对象的位置的两个坐标（4个数字的元组）；
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        # 移动正方形
        self.canvas.move(self.rect, base_action[0], base_action[1])
        # 下一个状态
        s_ = self.canvas.coords(self.rect)
        if s_ == self.canvas.coords(self.oval):
            # 到达了黄色点
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell), self.canvas.coords(self.hell2)]:
            # 到达了黑色点
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        # 渲染，
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


if __name__ == "__main__":
    env = Maze()
    env.after(100, update)
    env.mainloop()
