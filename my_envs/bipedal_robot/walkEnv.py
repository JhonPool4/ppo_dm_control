from dm_control.suite import base
from dm_control.utils import io as resources
from dm_control.utils import rewards
from dm_control import mujoco

import numpy as np
from copy import copy
import os

# reward parameters: 
_STANDUP_LOWER_TRESHOLD = 1.5 # meters
_STANDUP_UPPER_TRESHOLD = 2.5 # meters
_MAX_STANDUP_ERROR = 0.6 # meters

_FOOT_XLENGTH = 0.2 # meters
_FOOT_YLENGHT = 0.1 # meters

_VELOCITY_LOWER_TRESHOLD = 1.3 # meters/seconds
_VELOCITY_UPPER_TRESHOLD = 1.5 # meters/seconds
_MAX_VELOCITY_ERROR = 0.5 # meters/seconds


# initial states: disturbance
_ANGLE_DISTURBANCE = np.deg2rad(1) # radians

# neural network parameters
_ACTION_DIMENSION = 7
_OBSERVATION_DIMENSION = 22

#path of rl environemnt
_RL_DIR='/home/jhon/reinforcement_learning/ppo_dm_control/my_envs/bipedal_robot'
_MODEL_FILENAME='bipedal_robot.xml'    

# color
_MATERIALS = "body"
_DEFAULT = "body_default"
_HIGHLIGHT = "body_highlight"


def get_model(env_path, model_filename):
  """Returns a tuple containing the model XML string and a dict of assets."""
  return resources.GetResource(os.path.join(env_path, model_filename))

class Physics(mujoco.Physics):
    def body_xycom_pos(self):
        return np.array([self.named.data.sensordata['body_com_sensor'][0], self.named.data.sensordata['body_com_sensor'][1]])
    def lfoot_xycom_pos(self):
        return np.array([self.named.data.sensordata['l_foot_com_sensor'][0], self.named.data.sensordata['l_foot_com_sensor'][1]])
    def rfoot_xycom_pos(self):
        return np.array([self.named.data.sensordata['r_foot_com_sensor'][0], self.named.data.sensordata['r_foot_com_sensor'][1]])
    
    def body_xcom_vel(self):
        return np.array([self.named.data.sensordata['body_linvel_sensor'][0]])
    def joint_state_pos(self):
        return self.named.data.sensordata[[ 'trunk_pos_sensor', \
                                            'waist_pos_sensor', \
                                            'l_hip_pos_sensor', \
                                            'l_knee_pos_sensor', \
                                            'l_ankle_pos_sensor', \
                                            'r_hip_pos_sensor', \
                                            'r_knee_pos_sensor', \
                                            'r_ankle_pos_sensor']]
    def joint_state_vel(self):
        return self.named.data.sensordata[[ 'trunk_vel_sensor', \
                                            'waist_vel_sensor', \
                                            'l_hip_vel_sensor', \
                                            'l_knee_vel_sensor', \
                                            'l_ankle_vel_sensor', \
                                            'r_hip_vel_sensor', \
                                            'r_knee_vel_sensor', \
                                            'r_ankle_vel_sensor']]                                            
    def trunk_zcom_pos(self):
        return np.array([self.named.data.xpos['walker_robot', 'z']])
    def foot_force_ht(self):
        return self.named.data.sensordata[['l_foot_heel_force_sensor','l_foot_toe_force_sensor', 'r_foot_heel_force_sensor','r_foot_toe_force_sensor']]


class Walk(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)
    
    def initialize_episode(self, physics):
        """
        physics.named.data.qpos['root_xtras']=0.0
        physics.named.data.qpos['root_yrot']=np.deg2rad(0) + np.random.uniform(-_ANGLE_DISTURBANCE, _ANGLE_DISTURBANCE)
        physics.named.data.qpos['root_ztras']=0.0
        physics.named.data.qpos['waist'] = np.deg2rad(0) + np.random.uniform(-_ANGLE_DISTURBANCE, _ANGLE_DISTURBANCE)
        physics.named.data.qpos['l_hip']= np.deg2rad(-90) + np.random.uniform(-_ANGLE_DISTURBANCE, _ANGLE_DISTURBANCE)
        physics.named.data.qpos['l_knee']= np.deg2rad(50) + np.random.uniform(-_ANGLE_DISTURBANCE, _ANGLE_DISTURBANCE)
        physics.named.data.qpos['l_ankle']= np.deg2rad(0) + np.random.uniform(-_ANGLE_DISTURBANCE, _ANGLE_DISTURBANCE)
        physics.named.data.qpos['r_hip']= np.deg2rad(0) + np.random.uniform(-_ANGLE_DISTURBANCE, _ANGLE_DISTURBANCE)
        physics.named.data.qpos['r_knee']= np.deg2rad(30) + np.random.uniform(-_ANGLE_DISTURBANCE, _ANGLE_DISTURBANCE)
        physics.named.data.qpos['r_ankle']= np.deg2rad(-30) + np.random.uniform(-_ANGLE_DISTURBANCE, _ANGLE_DISTURBANCE)            
        """ 
        super().initialize_episode(physics)

    def get_observation(self, physics):
        # bodycom_dx: 1
        # joint_states: 16
        # trunkzcom: 1
        # force_ht: 4 (l, r)
        # total: 22
        return np.concatenate(( physics.body_xcom_vel(), \
                                physics.joint_state_pos(), \
                                physics.joint_state_vel(), \
                                physics.trunk_zcom_pos(), \
                                physics.foot_force_ht() ))   

    def get_reward(self, obs):
        # trunk: angle and  zcom
        trunk_zcom = obs[17]   
        
        # body COM: x-axis velocity
        body_dxcom = obs[0]

        # reward system: to encourage stand up
        stand_reward = rewards.tolerance(trunk_zcom,
                                            bounds=(_STANDUP_LOWER_TRESHOLD, _STANDUP_UPPER_TRESHOLD),
                                            margin=_MAX_STANDUP_ERROR/2,
                                            value_at_margin=0.5,
                                            sigmoid='gaussian') 
        
        # reward system: to encourage forward motion and velocity tracking
        forward_reward = rewards.tolerance( body_dxcom, 
                                            bounds=(_VELOCITY_LOWER_TRESHOLD,_VELOCITY_UPPER_TRESHOLD),
                                            margin=_MAX_VELOCITY_ERROR/2,
                                            value_at_margin=0.5,
                                            sigmoid='linear')

        # reward system: static stability 
        #stability_reward = np.where(self.inside_support_polygon(l_foot, r_foot, body_com), 1.0, 0.0)                                            
        
        # final reward system: standup + forward motion + static stability + good posture
        cost = (1 + 5*stand_reward)*(1 + 3*forward_reward)   
        return cost/(6*4)


    def inside_support_polygon(sefl, l_foot, r_foot, body_com):
        """
        @info indicates if body_com is inside support polygon
        """
        # corners
        corners= [  np.array([l_foot[0] - _FOOT_XLENGTH, l_foot[1]]),\
                    np.array([l_foot[0] + _FOOT_XLENGTH, l_foot[1]]), \
                    np.array([r_foot[0] + _FOOT_XLENGTH, r_foot[1]]), \
                    np.array([r_foot[0] - _FOOT_XLENGTH, r_foot[1]]), \
                    np.array([l_foot[0] - _FOOT_XLENGTH, l_foot[1]]) ]

        #print(f"corners: {corners} \n")
        #print(f"body: {body_com} \n")

        # eval static statiblity 
        angle=0.0
        for i in range(len(corners)-1):
            c1=corners[i]
            c2=corners[i+1]
            A=np.linalg.norm(c1-c2)
            B=np.linalg.norm(c1-body_com)
            C=np.linalg.norm(c2-body_com)
            angle+=np.arccos((B*B + C*C - A*A)/(2*B*C))
        
        if abs(angle-2*np.pi)<=np.deg2rad(10):
            #inside
            #print("inside")
            return True
        else:
            return False

class CustomEnv():
    def __init__(self,
                 physics,
                 task,
                 simulation_time):
        self._physics=physics
        self._task=task
        self._step_limit=simulation_time/(self._physics.timestep())
        # initial values
        self._step_count=0
        self.action_dim=_ACTION_DIMENSION
        self.observation_dim=_OBSERVATION_DIMENSION

    def reset(self):
        # initial values
        self._step_count=0

        # initial states/configuration of physics
        with self._physics.reset_context():
            self._task.initialize_episode(self._physics)
        # observation of initial states
        observation=self._task.get_observation(self._physics)
        
        return observation  

    def step(self, action):
        # apply action
        self._task.before_step(action, self._physics)
        self._physics.step()
        # observation of initial states
        observation=self._task.get_observation(self._physics)
        # reward
        reward = self._task.get_reward(observation)
        # normalization of the observated states
        norm_obs = (observation - observation.mean())/observation.std()
        # colors
        self._set_reward_colors(self._physics, reward)          
        # steps
        self._step_count += 1
        
        # body COM: x-axis position
        body_xycom = self._physics.body_xycom_pos()
        # foot COM: xy
        l_foot = self._physics.lfoot_xycom_pos()
        r_foot = self._physics.rfoot_xycom_pos()


        if self._step_count>=self._step_limit: # end episode: max steps
            return norm_obs, 0.0, True, {}

        if not np.where(self._task.inside_support_polygon(l_foot, r_foot, body_xycom), 1.0, 0.0):
            return norm_obs, 0.0, True, {}

        if abs(observation[1])>=abs(np.deg2rad(120)):
            return norm_obs, 0.0, True, {}

        if observation[17]<(_STANDUP_LOWER_TRESHOLD-_MAX_STANDUP_ERROR):
            return norm_obs, 0.0, True, {}

        else:
            return norm_obs, reward, False, {} # learning           

    def _set_reward_colors(self, physics, reward):
        assert 0.0 <= reward <= 1.0
        colors = physics.named.model.mat_rgba
        default = colors[_DEFAULT]
        highlight = colors[_HIGHLIGHT]
        blend_coef = reward ** 2  # Better color distinction near high rewards.
        colors[_MATERIALS] = blend_coef * highlight + (1.0 - blend_coef) * default


def walkEnv(time_limit=5, random=None, env_kwargs=None):
    physics =  Physics.from_xml_string(get_model(_RL_DIR, _MODEL_FILENAME))
    task = Walk(random=random)
    environment_kwargs = env_kwargs or {}
    return CustomEnv(physics, task, time_limit, **environment_kwargs)