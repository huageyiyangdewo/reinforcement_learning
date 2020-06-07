import time

import pandas as pd
import numpy as np


"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

learning from: morvanzhou
"""

np.random.seed(2)  # reproducible 为了每次生成的随机数都是一样的


N_STATES = 6  # the length of the 1 dimensional world
ACTIONS = ['left', 'right']   # available actions
EPISODES = 0.9  # greedy police  贪婪度
ALPHA = 0.1  # learning rate  学习率
GAMMA = 0.9  # discount factor 奖励衰减值
MAX_EPISODE = 13  # maximum episodes 最大回合数
FRESH_TIME = 0.3  # fresh time for one move


def build_q_table(n_states, actions):
    '''
    对于 tabular Q learning, 我们必须将所有的 Q values (行为值) 放在 q_table 中, 更新 q_table 也是在更新他的行为准则.
     q_table 的 index 是所有对应的 state (探索者位置), columns 是对应的 action (探索者行为).
    :param n_states:
    :param actions:
    :return:
    '''

    table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)
    return table


def choose_action(state, q_table):
    '''
    接着定义探索者是如何挑选行为的. 这是我们引入 epsilon greedy 的概念. 因为在初始阶段, 随机的探索环境,
     往往比固定的行为模式要好, 所以这也是累积经验的阶段, 我们希望探索者不会那么贪婪(greedy).
     所以 EPSILON 就是用来控制贪婪程度的值. EPSILON 可以随着探索时间不断提升(越来越贪婪),
     不过在这个例子中, 我们就固定成 EPSILON = 0.9, 90% 的时间是选择最优策略, 10% 的时间来探索.
    :param state:
    :param q_table:
    :return:
    '''
    state_actions = q_table.iloc[state, :]  # 选出这个 state 的所有 action 值
    # 非贪婪 or 或者这个 state 还没有探索过
    if np.random.uniform() > MAX_EPISODE or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()   # 贪婪模式

    return action_name


def get_env_feedback(s, a):
    '''
    做出行为后, 环境也要给我们的行为一个反馈, 反馈出下个 state (S_) 和 在上个 state (S) 做出 action (A) 所得到的
    reward (R). 这里定义的规则就是, 只有当 o 移动到了 T, 探索者才会得到唯一的一个奖励, 奖励值 R=1, 其他情况都没有奖励.
    :param s: 上个 state
    :param a: 做出 action
    :return:
    '''
    # This is how agent will interact with the environment
    if a == 'right':   # move right
        if s == N_STATES - 2:  # env_list的长度为6，下标从0开始，所以 -2
            s_ = 'terminal'
            r = 1
        else:
            s_ = s + 1
            r = 0
    else:
        r = 0
        if s == 0:  # 到达最左边，无法再移动
            s_ = s
        else:
            s_ = s - 1

    return s_, r


def update_env(s, episode, step_counter):
    '''
    环境的更新
    :param s:
    :param episode:
    :param step_counter:
    :return:
    '''
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if s == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[s] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    '''RL 方法都在这里体现'''

    # 初始化q table
    q_table = build_q_table(N_STATES, ACTIONS)

    for episode in range(MAX_EPISODE):  # 回合
        step_counter = 0
        s = 0  # 初始化位置
        is_terminated = False  # 是否回合结束
        update_env(s, episode, step_counter)  # 环境更新

        while not is_terminated:
            # 选行为
            a = choose_action(s, q_table)
            # 实施行为并得到环境的反馈
            s_, r = get_env_feedback(s, a)
            # 估算的(状态-行为)值
            q_predict = q_table.loc[s, a]
            if s_ != 'terminal':
                #  实际的(状态-行为)值 (回合没结束)
                q_target = r + GAMMA * q_table.iloc[s_, :].max()
            else:
                #  实际的(状态-行为)值 (回合结束)
                q_target = r
                is_terminated = True  # 介绍

            #  q_table 更新
            q_table.loc[s, a] += ALPHA * (q_target - q_predict)
            # 探索者移动到下一个 state
            s = s_

            update_env(s, episode, step_counter + 1)  # 环境更新
            step_counter += 1
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
