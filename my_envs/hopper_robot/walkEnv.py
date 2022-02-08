from dm_control.suite import base
from dm_control.utils import io as resources
from dm_control import mujoco
from dm_control.utils import rewards

import numpy as np
import os

# reward parameters
_STANDUP_LOWER_TRESHOLD = 1.15 # meters
_STANDUP_UPPER_TRESHOLD = 2.0 # meters
_MAX_STANDUP_ERROR = 0.5 # meters

_POSTURAL_LOWER_TRESHOLD = np.deg2rad(-30) # degrees
_POSTURAL_UPPER_TRESHOLD = np.deg2rad(30) # degrees
_MAX_POSTURAL_ERROR = np.deg2rad(20) # degrees

_VELOCITY_LOWER_TRESHOLD = 1.3 # meters/seconds
_VELOCITY_UPPER_TRESHOLD = 1.5 # meters/seconds
_MAX_VELOCITY_ERROR = 0.5 # meters/seconds

# initial states: disturbance
_ANGLE_DISTURBANCE = np.deg2rad(0) # radians

# neural network parameters
_ACTION_DIMENSION = 4
_OBSERVATION_DIMENSION = 19

#path of rl environemnt
_RL_DIR='/home/jhon/reinforcement_learning/ppo_dm_control/my_envs/hopper_robot'
_MODEL_FILENAME='hopper.xml'    


def get_model(env_path, model_filename):
  """Returns a tuple containing the model XML string and a dict of assets."""
  return resources.GetResource(os.path.join(env_path, model_filename))


class Physics(mujoco.Physics):
    def body_com_pos(self):
        return self.named.data.sensordata['body_com_sensor']
    def body_com_vel(self):
        return self.named.data.sensordata['body_linvel_sensor']
    def trunk_ang_pos(self):
        return self.named.data.sensordata['trunk_angle_sensor']
    def trunk_zcom_pos(self):
        return np.array([self.named.data.xpos['walker_robot', 'z']])
    def trunk_imu(self):
        return self.named.data.sensordata[['trunk_linaccel_sensor', 'trunk_angaccel_sensor']]
    def foot_com_pos(self):
        return self.named.data.sensordata['foot_com_sensor']
    def foot_force_ht(self):
        return self.named.data.sensordata[['foot_heel_force_sensor','foot_toe_force_sensor']]

class Walk(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)
    
    def initialize_episode(self, physics):
        physics.named.data.qpos['root_xtras']=0.0
        physics.named.data.qpos['root_yrot']=np.deg2rad(-90)+ np.random.uniform(-_ANGLE_DISTURBANCE, _ANGLE_DISTURBANCE)
        physics.named.data.qpos['root_ztras']=0.0
        physics.named.data.qpos['hip']= np.deg2rad(-90) + np.random.uniform(-_ANGLE_DISTURBANCE, _ANGLE_DISTURBANCE)
        physics.named.data.qpos['knee']= np.deg2rad(50) + np.random.uniform(-_ANGLE_DISTURBANCE, _ANGLE_DISTURBANCE)
        physics.named.data.qpos['ankle']= np.deg2rad(30) + np.random.uniform(-_ANGLE_DISTURBANCE, _ANGLE_DISTURBANCE)
        
        super().initialize_episode(physics)

    def get_observation(self, physics):
        # bodycompos: 3
        # bodycomvel: 3
        # angpos: 1
        # zcom: 1
        # imu: 6
        # footcom: 3
        # force_ht: 2
        # total: 19        
        return np.concatenate(( physics.body_com_pos(), \
                                physics.body_com_vel(), \
                                physics.trunk_ang_pos(), \
                                physics.trunk_zcom_pos(), \
                                physics.trunk_imu(), \
                                physics.foot_com_pos(), \
                                physics.foot_force_ht() )) 

    def get_reward(self, obs):
        # trunk: angle and  zcom
        trunk_angle = obs[6]
        trunk_zcom = obs[7]    

        # body COM: velocity
        body_dxcom = obs[3]
        # reward system: to encourage stand up
        stand_reward = rewards.tolerance(trunk_zcom,
                                            bounds=(_STANDUP_LOWER_TRESHOLD, _STANDUP_UPPER_TRESHOLD),
                                            margin=_MAX_STANDUP_ERROR/2,
                                            value_at_margin=0.5,
                                            sigmoid='gaussian') 
        
        standup_reward = rewards._sigmoids((_STANDUP_UPPER_TRESHOLD-trunk_zcom)/_STANDUP_UPPER_TRESHOLD,
                                            value_at_1=1e-10,
                                            sigmoid='linear')  


        # reward system: to encourage forward motion and velocity tracking
        forward_reward = rewards.tolerance( body_dxcom, 
                                            bounds=(_VELOCITY_LOWER_TRESHOLD,_VELOCITY_UPPER_TRESHOLD),
                                            margin=_MAX_VELOCITY_ERROR/2,
                                            value_at_margin=0.5,
                                            sigmoid='linear')
        
        # reward system: to encourage good posture      
        posture_reward = rewards.tolerance(trunk_angle,
                                            bounds=(_POSTURAL_LOWER_TRESHOLD, _POSTURAL_UPPER_TRESHOLD),
                                            margin=_MAX_POSTURAL_ERROR/2,
                                            value_at_margin=0.5,
                                            sigmoid='gaussian')

        # final reward system: standup * forward * postural       
        return (1 + 3*stand_reward + 0.5*standup_reward)*(1 + 5*forward_reward)*(1 + 0.1*posture_reward)        

class CustomEnv():
    def __init__(self,
                 physics,
                 task,
                 simulation_time):
        """
        - physics: Instance of Physics
        - task: Instante of Task
        - simulation_time: maximum simulation time of each episode [seconds]
        """
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
        self._task.after_step(self._physics)      
        
        # observation of initial states
        observation=self._task.get_observation(self._physics)
        # reward
        reward = self._task.get_reward(observation)          
        # steps
        self._step_count += 1

        if self._step_count>=self._step_limit: # end episode: max steps
            return observation, 0.0, True, {}
        
        else:
            return observation, reward, False, {} # learning   
        


def walkEnv(time_limit=5, random=None, env_kwargs=None):
    
    physics =  Physics.from_xml_string(get_model(_RL_DIR, _MODEL_FILENAME))
    task = Walk(random=random)
    environment_kwargs = env_kwargs or {}
    return CustomEnv(physics, task, time_limit, **environment_kwargs)