# Azure ML Core imports
import azureml.core
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget

# Azure ML Reinforcement Learning imports
from azureml.contrib.train.rl import ReinforcementLearningEstimator, Ray

ws = Workspace.from_config()

compute_name = 'scaled-nodes-1'

if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print(f'found big compute target. just use it {compute_name}')
    else:
        print ("ERROR: couldn't find compute target")
        exit(1)
else:
    print ("ERROR: couldn't find compute target")
    exit(1)

experiment_name='testing-docker-env'

exp = Experiment(workspace=ws, name=experiment_name)

script_params = {
    "--run": "DQN",
    "--env": "CartPole-v1",
    "--stop": '\'{"episode_reward_mean": 18, "time_total_s": 3600}\'',
    "--ray-address": "auto"
}

#myenv = Environment.from_docker_image('myenv', "seeloz/ray-sample:latest")
#myenv.python.user_managed_dependencies = True

estimator = ReinforcementLearningEstimator(

    #environment=myenv,

    source_directory='./ray-sample/',

    # Python script file
    entry_script="main.py",

    # Parameters to pass to the script file
    # Defined above.
    script_params=script_params,

    # The Azure ML compute target set up for Ray head nodes
    compute_target=compute_target,

    # Pip packages
    # pip_requirements_file='requirements.txt',

    # GPU usage
    use_gpu=False,

    # RL framework. Currently must be Ray.
    rl_framework=Ray(),

    # Ray worker configuration defined above.
    # worker_configuration=worker_conf,

    # How long to wait for whole cluster to start
    cluster_coordination_timeout_seconds=3600,

    # Maximum time for the whole Ray job to run
    max_run_duration_seconds=3600*20

    # Allow the docker container Ray runs in to make full use
    # of the shared memory available from the host OS.
    # shm_size=24*1024*1024*1024
)

run = exp.submit(config=estimator)
