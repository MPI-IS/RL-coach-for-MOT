import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import interp1d

from tensorflow import keras
from tensorflow.keras import layers

from IPython import embed
import os



from rl_coach import logger
from rl_coach.core_types import EnvironmentSteps
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.spaces import DiscreteActionSpace, ImageObservationSpace, VectorObservationSpace, StateSpace,BoxActionSpace
from rl_coach.environments.environment import Environment, EnvironmentParameters, LevelSelection
from typing import Union

from rl_coach.filters.filter import NoInputFilter, NoOutputFilter

#needed for TD3
class DummyClass():
    def __init__(self):
        self._max_episode_steps = 0

class ContMOTEnvironmentParameters(EnvironmentParameters):
    def __init__(self):
        super().__init__()
        self.frame_skip = 1
        self.random_initialization_steps = 1
        self.default_input_filter = NoInputFilter()
        self.default_output_filter = NoOutputFilter()
        self.var_det = 1
        self.NEv = 1
        self.custom_param = 1

    @property
    def path(self):
        return 'rl_coach.environments.ContMOT_environment:ContMOTEnvironment'



class ContMOTEnvironment(Environment):

   
    def __init__(self, level: LevelSelection, frame_skip: int, visualization_parameters: VisualizationParameters,
                 seed: Union[None, int]=None, human_control: bool=False,
                 custom_reward_threshold: Union[int, float]=None, **kwargs):

        super().__init__(level, seed, frame_skip, human_control, custom_reward_threshold, visualization_parameters)


        os.mkdir( kwargs['experiment_path'] + "/npz")
        #embed()
        
        self.img_size = 50
        self.Ttot = 25
        self.env = DummyClass()                 #needed for TD3
        self.env._max_episode_steps = self.Ttot #needed for TD3

        
        self.var_det = kwargs['var_det']
        self.custom_param = kwargs['custom_param']
        
        
        self.NEv = kwargs['NEv']
        self.Reward_rescale = kwargs['Reward_rescale']
		
        self.det_hist = list()

        
        self.evalshot_idx = 1
        
        self._restart_environment_episode()
        
        self.state_space = StateSpace({})
        self.state_space['observation'] = ImageObservationSpace(shape = (self.img_size, self.img_size),high=255) 
        self.state_space['measurements'] = VectorObservationSpace(shape = (2),low = -2, high=2)
		
        
        self.action_space = BoxActionSpace(shape=1, low=0, high=1, descriptions=["detuning"])
        
        # load experimental model:
        exp_model_path = 'exp_MOT_dataset/FluoImgs7_results/'
        
        
        self.MOT_img_gen =  keras.models.load_model(exp_model_path + 'MOT_fluo_img_generator.h5')
        
        
        self.L_exp = np.squeeze(np.load(exp_model_path + 'LUT_L.npy'))
        self.det_L = np.squeeze(np.load(exp_model_path + 'det_L.npy'))
        N_max = np.squeeze(np.load(exp_model_path + 'N_max.npy'))
        
        self.L_exp = self.L_exp/N_max
        self.det_L = - self.det_L

        self.L_intrp = interp1d(self.det_L,self.L_exp,'cubic',bounds_error=False,fill_value =0)
        
        
        self.T_exp = np.squeeze(np.load(exp_model_path + 'LUT_T.npy'))
        self.det_T = np.squeeze(np.load(exp_model_path + 'det_T.npy'))
        
        self.T_exp = 0.1*self.T_exp/self.T_exp[-1] 
       
        self.det_T = - self.det_T
        
        self.logT_intrp = interp1d(self.det_T,np.log10(self.T_exp))
        
        
        
        
        
        self.native_rendering = True
		
        self.evaluation = False
        # initialize the state by getting a new state from the environment
        self.reset_internal_state(True)


    def _restart_environment_episode(self, force_environment_reset=False) -> None:
        """
        Restarts the simulator episode
        :param force_environment_reset: Force the environment to reset even if the episode is not done yet.
        :return: None
        """
        
        self.Natoms = 0
        self.Temp = 1
        self.t = 0
        self.img = np.zeros([self.img_size,self.img_size])
        


        self.done = False
        self.reward = 0

        self.N0 = random.uniform(1,20) #goal for the number of atoms at the end
        self.T0 = random.uniform(0.1,1.1) #goal for the temperature at the end
        
        self.det = -20 # has no real effect, as it is overwritten by the first action
        #self.det_off = self.var_det*random.uniform(-5, 10)
        self.det_off = np.floor(((self.episode_idx+1) % 16)/2)*2 - 4
        
    

		
        self.det_hist.clear()
        
        return 0 



    
    def loading(self,det):
	"""
	simulate the loading of the MOT
	"""
	
       
        return max((0,self.L_intrp(det)*random.gauss(1,0.2))) #

    
    def temperature(self,det):
	"""
	simulate the temperature of the MOT
	"""

        logT_intrp = self.logT_intrp
        if det>=max(self.det_T):
            return self.T_exp[0]
        elif det<=np.min(self.det_T):
            return self.T_exp[-1]
        else:
            return 10**logT_intrp(det)


    def draw_MOT_img(self):
	"""
	generate the fluorescence image using the CNN model
	"""
        
        det2exp = self.det + self.det_off
        self.img = self.MOT_img_gen.predict(np.expand_dims((self.Natoms/self.Ttot, -det2exp/50),axis=0))*255
        self.img = self.img.squeeze()
        if self.Natoms<0.02 :
            self.img = self.img*0
        self.img = self.img.clip(min = 0)
        
    

    def _render(self):
			
		
		
     
    def get_rendered_image(self) -> np.ndarray:
        """
        Return a numpy array containing the image that will be rendered to the screen.
        This can be different from the observation. For example, mujoco's observation is a measurements vector.
        :return: numpy array containing the image that will be rendered to the screen
        """
        
        return self.img



    def _take_action(self, action):
        """
	setting the new control parameter and letting the MOT evolve
	"""
        
            
        if not self.done:
            self.t +=1
            
	    #this implements the step during the evaluation, change offset at timestep 15
            if (self.evalshot_idx > self.NEv/2) and  (self.t == np.round(self.Ttot*3/5)): 
                self.det_off = self.det_off + random.choice((-1, 1))*5
                

            self.det = -action[0]*50
            det2exp = self.det + self.det_off
            
	    
            if det2exp<-2.5:
            
                self.Natoms += self.loading(det = det2exp, det_opt = self.det_opt)
               
                self.Temp = self.temperature(self.Natoms,det2exp)
	    
	    #remove atoms when the detuning is too close to resonance
            else:
                self.Natoms = 0
                self.Temp = 5
        
            self.det_hist.append(self.det)


			
        
            
    def _update_state(self) -> None:
        """
        Updates the state from the environment.
        Should update self.observation, self.reward, self.done, self.measurements and self.info
        :return: None
        """

        
        self.draw_MOT_img()
        self.state['observation'] = self.img
        self.state['measurements'] = self.t/self.Ttot, -self.det/50
        #self.state['measurements'] = self.t/self.Ttot, -self.det/50, self.N0/25, self.T0

        
        
        if self.t>=self.Ttot :
            self.done = True
            self.reward = self.Natoms/self.Temp/self.Ttot 
            self.reward = (self.gauss(self.Natoms,self.N0,1)/self.Temp/self.Ttot
            if self.Natoms<0.1:
                self.reward = 0
        
      


    
    

    def dump_video_of_last_episode(self):
        exp_path = logger.experiment_path
        file_name = 'episode-{:05d}_shot-{:02d}'.format(self.episode_idx+1, self.evalshot_idx)
        frame_skipping = 1
        np.savez(exp_path+'/npz/'+file_name, Reward = self.reward, Temperature = self.Temp, Atomcount = self.Natoms, Detuning = self.det_hist,  Detuning_offset = self.det_off, N0 = self.N0, T0 = self.T0)
        np.savez(exp_path+'/npz/fImg_'+file_name, fImgs = np.uint8(self.last_episode_images[::frame_skipping]))
         
        self.evalshot_idx += 1
        if self.evalshot_idx > self.NEv:
            self.evalshot_idx = 1
            


        
    

    def game_over(self):
        return self.done


    def lives(self):
        if self.done:
            return 0
        else:
            return 1
            
    def gauss(self,x,mu,sigma):
        return np.exp(-(x-mu)**2/(2*sigma**2))
            
      
        

if __name__ == '__main__':


 
    MOT = MOT_env()
