"""Public RL environment using exactly the official RL physics and rewards."""
import gymnasium as gym
import numpy as np
from driving import DrivingController
from gui import make_driving_env
from rl_rules import RL_DRIVING
from sdc_wrapper import SDC_Wrapper
from track import TRACK_SEED

ACTIONS=((0,0,0),(0,.5,0),(0,0,.8),(-1,0,0),(1,0,0),(-1,.5,0),(1,.5,0),(-1,0,.8),(1,0,.8))


class RLDrivingEnv(gym.Wrapper):
    def __init__(self, display=False, max_steps=30000, reward_function=None):
        if type(max_steps) is not int or max_steps<1:
            raise ValueError('max_steps must be a positive integer')
        super().__init__(SDC_Wrapper(make_driving_env(display=display,unlimited=True,assignment='rl')))
        self.action_space=gym.spaces.Discrete(len(ACTIONS))
        self.controller=DrivingController(RL_DRIVING,50)
        self.reward_function=reward_function
        self.max_steps=max_steps
        self.steps=0

    def reset(self, *, seed=None, options=None):
        self.controller.reset();self.steps=0
        obs,info=self.env.reset(seed=TRACK_SEED if seed is None else seed,options=options)
        self.unwrapped.metrics.reward_function=self.reward_function
        return obs,dict(info,**self.unwrapped.metrics.result())

    def step(self,action):
        if not self.action_space.contains(action):raise ValueError('Action must be an integer from 0 to 8')
        speed=np.linalg.norm(self.unwrapped.car.hull.linearVelocity)
        obs,reward,done,truncated,info=self.env.step(self.controller.apply(ACTIONS[int(action)],speed))
        self.steps+=1
        return obs,reward,done,truncated or self.steps>=self.max_steps,info
