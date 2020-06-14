"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
"""

from Sarsa_maze.maze_env import Maze
from Sarsa_maze.RL_brain import SarsaTable


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()
        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation
            observation = observation_
            action = action_

            # break while loop when end of this episode
            print(done)
            if done:
                break

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    # 每一百毫秒调用一次update
    env.after(100, update)
    env.mainloop()
