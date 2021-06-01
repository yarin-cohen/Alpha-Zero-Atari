import gym
from gym.core import Wrapper
from collections import namedtuple

# a container for get_result function below. Works just like tuple, but prettier
ActionResult = namedtuple(
    "action_result", ("snapshot", "observation", "reward", "is_done", "info"))
"""this class wraps the original gym atari environment. it keeps tabs on various snapshots of the game"""
"""example of usage: env = WithSnapshots(gym.make("Breakout-ram-v0").env)"""


class WithSnapshots(Wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pics = {}
        self.curr_pic = []
        self.num_steps = 0
        self.img_shape = self.env.unwrapped._get_image().shape

    def reset(self):
        obs = self.env.reset()
        self.curr_pic = self.unwrapped._get_image()
        self.num_steps = 0
        return obs

    def get_snapshot(self):
        s = self.env.ale.cloneState()
        self.pics[s] = self.unwrapped._get_image()
        return s

    def restore_snapshot(self, s):
        self.env.ale.restoreState(s)
        self.curr_pic = self.pics[s]

    def step(self, action):
        result = self.env.step(action)
        self.curr_pic = self.unwrapped._get_image()
        return result

    def get_result(self, snapshot, action):
        """
        A convenience function that
        - loads snapshot,
        - commits action via self.step,
        - and takes snapshot again :)

        :returns: next snapshot, next_observation, reward, is_done, info

        Basically it returns next snapshot and everything that env.step would have returned.
        """
        # self.env.reset()
        self.restore_snapshot(snapshot)
        l1 = self.unwrapped.ale.lives()
        new_s, r, done, info = self.step(action)
        l2 = self.unwrapped.ale.lives()
        if l2 < l1:
            r -= 3
        new_snap = self.get_snapshot()

        return ActionResult(new_snap,  # fill in the variables
                            new_s,
                            r, done, info)