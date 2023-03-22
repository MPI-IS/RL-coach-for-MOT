from rl_coach.agents.ddpg_agent import DDPGAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, MiddlewareScheme, EmbedderScheme
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.architectures.layers import Dense
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.environments.ContMOT_environment import ContMOTEnvironmentParameters
from rl_coach.graph_managers.graph_manager import ScheduleParameters, SimpleSchedule
from rl_coach.core_types import EnvironmentSteps, TrainingSteps, EnvironmentEpisodes, SelectedPhaseOnlyDumpFilter, RunPhase
from rl_coach.schedules import LinearSchedule, ConstantSchedule
from rl_coach.memories.non_episodic.prioritized_experience_replay import PrioritizedExperienceReplayParameters

	
####################
# Graph Scheduling #
####################

schedule_params = SimpleSchedule()

number_evals = 10
schedule_params.evaluation_steps = EnvironmentEpisodes(number_evals)
schedule_params.heatup_steps = EnvironmentSteps(25000) 
schedule_params.improve_steps = TrainingSteps(1500000) 
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(200)


#########
# Agent #
#########

agent_params = DDPGAgentParameters()

#agent_params.network_wrappers['critic'].l2_regularization = 1e-2

agent_params.network_wrappers['actor'].middleware_parameters.scheme = [Dense(200),Dense(200)]
agent_params.network_wrappers['critic'].middleware_parameters.scheme = [Dense(200),Dense(200)]

agent_params.network_wrappers['actor'].input_embedders_parameters =  {'observation': InputEmbedderParameters(),'measurements': InputEmbedderParameters()} 
agent_params.network_wrappers['critic'].input_embedders_parameters =  {'observation': InputEmbedderParameters(),'measurements': InputEmbedderParameters(),'action': InputEmbedderParameters(scheme=EmbedderScheme.Shallow)} 

agent_params.network_wrappers['actor'].learning_rate = 0.0001
agent_params.network_wrappers['critic'].learning_rate = 0.001


agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(25)

agent_params.exploration.sigma = 0.2


from rl_coach.filters.filter import InputFilter
from rl_coach.filters.observation.observation_stacking_filter import ObservationStackingFilter
from rl_coach.filters.observation.observation_to_uint8_filter import ObservationToUInt8Filter


MOTInputFilter = InputFilter(is_a_reference_filter=True)
MOTInputFilter.add_observation_filter('observation', 'to_uint8', ObservationToUInt8Filter(0, 255))
MOTInputFilter.add_observation_filter('observation', 'stacking', ObservationStackingFilter(4))


###############
# Environment #
###############
env_params = ContMOTEnvironmentParameters()
env_params.default_input_filter = MOTInputFilter()
env_params.var_det = 1
env_params.NEv = number_evals
env_params.custom_param = 5

########
# Test #
########
preset_validation_params = PresetValidationParameters()

################
# Visualization#
################
vis_params=VisualizationParameters()
vis_params.video_dump_filters  = [SelectedPhaseOnlyDumpFilter(RunPhase.TEST)]

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)
