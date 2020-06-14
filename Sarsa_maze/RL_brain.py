"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
思维决策部分
"""

import numpy as np
import pandas as pd


class RL(object):

    def __init__(self, actions, learning_rate, reward_decay, e_greedy):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        '''
        选行为
        :param observation:
        :return:
        '''
        self.check_state_exist(observation)

        # 选择 Q value 最高的 action
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            # 同一个 state, 可能会有多个相同的 Q action value, 所以我们乱序一下
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def check_state_exist(self, state):
        '''
        检查 state 是否存在
        :param state:
        :return:
        '''
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )


# off-policy
class QLearningTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        '''
        学习
        :param s:
        :param a:行为action
        :param r:奖励值
        :param s_:下一个state
        :return:
        '''
        self.check_state_exist(s_)
        # 估算的(状态-行为)值
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            #  实际的(状态-行为)值 (回合没结束)
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal

        # 神经网络中的更新方式, 学习率 * (真实值 - 预测值). 将判断误差传递回去, 有着和神经网络更新的异曲同工之处.
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update


class SarsaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        '''
        学习
        :param s:
        :param a:行为action
        :param r:奖励值
        :param s_:下一个state
        :param a_:下一个action
        :return:
        '''
        self.check_state_exist(s_)
        # 估算的(状态-行为)值
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            #  实际的(状态-行为)值 (回合没结束)
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal

        # 神经网络中的更新方式, 学习率 * (真实值 - 预测值). 将判断误差传递回去, 有着和神经网络更新的异曲同工之处.
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update