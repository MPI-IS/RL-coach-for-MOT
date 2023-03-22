#
# Copyright (c) 2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import time
import atexit 

from typing import Tuple, List

from rl_coach.base_parameters import AgentParameters, VisualizationParameters, TaskParameters, \
    PresetValidationParameters
from rl_coach.environments.environment import EnvironmentParameters, Environment
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
from rl_coach.graph_managers.graph_manager import GraphManager, ScheduleParameters
from rl_coach.level_manager import LevelManager
from rl_coach.utils import short_dynamic_import
from rl_coach.logger import screen
from rl_coach.core_types import StepMethod, RunPhase, EnvironmentSteps #TotalStepsCounter, RunPhase, PlayingStepsType, TrainingSteps, EnvironmentEpisodes,  Transition
    
from IPython import embed

class VVRLGraphManager(GraphManager):
    """
    VV RL graphManager is mainly a copy and paste of BasicRLGraphManager with some  modified methods of GraphManager
    A basic RL graph manager creates the common scheme of RL where there is a single agent which interacts with a
    single environment.
    """
    def __init__(self, agent_params: AgentParameters, env_params: EnvironmentParameters,
                 schedule_params: ScheduleParameters,
                 vis_params: VisualizationParameters=VisualizationParameters(),
                 preset_validation_params: PresetValidationParameters = PresetValidationParameters(),
                 name='vv_rl_graph'):
        super().__init__(name, schedule_params, vis_params)

        self.agent_params = agent_params
        self.env_params = env_params
        self.preset_validation_params = preset_validation_params
        self.agent_params.visualization = vis_params
        
        #atexit.register(self.save_memory_buffer)

        if self.agent_params.input_filter is None:
            if env_params is not None:
                self.agent_params.input_filter = env_params.default_input_filter()
            else:
                # In cases where there is no environment (e.g. batch-rl and imitation learning), there is nowhere to get
                # a default filter from. So using a default no-filter.
                # When there is no environment, the user is expected to define input/output filters (if required) using
                # the preset.
                self.agent_params.input_filter = NoInputFilter()
        if self.agent_params.output_filter is None:
            if env_params is not None:
                self.agent_params.output_filter = env_params.default_output_filter()
            else:
                self.agent_params.output_filter = NoOutputFilter()

    def _create_graph(self, task_parameters: TaskParameters) -> Tuple[List[LevelManager], List[Environment]]:
        # environment loading
        self.env_params.seed = task_parameters.seed
        self.env_params.experiment_path = task_parameters.experiment_path
        env = short_dynamic_import(self.env_params.path)(**self.env_params.__dict__,
                                                         visualization_parameters=self.visualization_parameters)

        # agent loading
        self.agent_params.task_parameters = task_parameters  # TODO: this should probably be passed in a different way
        self.agent_params.name = "agent"
        agent = short_dynamic_import(self.agent_params.path)(self.agent_params)

        # set level manager
        level_manager = LevelManager(agents=agent, environment=env, name="main_level")

        return [level_manager], [env]

    def log_signal(self, signal_name, value):
        self.level_managers[0].agents['agent'].agent_logger.create_signal_value(signal_name, value)

    def get_signal_value(self, signal_name):
        return self.level_managers[0].agents['agent'].agent_logger.get_signal_value(signal_name)

    def get_agent(self):
        return self.level_managers[0].agents['agent']

    def occasionally_save_checkpoint(self):
        # only the chief process saves checkpoints
        if self.task_parameters.checkpoint_save_secs \
                and time.time() - self.last_checkpoint_saving_time >= self.task_parameters.checkpoint_save_secs \
                and (self.task_parameters.task_index == 0  # distributed
                     or self.task_parameters.task_index is None  # single-worker
                     ):
            self.save_checkpoint()
            self.save_memory_buffer()
            
    def save_memory_buffer(self):
        self.get_agent().memory.save(self.task_parameters.checkpoint_save_dir + '/mem')
        screen.log_title("Saving the memory buffer to {}/mem".format(self.task_parameters.checkpoint_save_dir ))
        