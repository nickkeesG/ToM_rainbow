#from absl import app
#from absl import flags

import run_experiment

#FLAGS = flags.FLAGS

#flags.DEFINE_string('gin_file', None,
#                    'Path to gin configuration file')

def launch_experiment():
    #if FLAGS.gin_file == None:
    #    raise ValueError('--gin_file is None: please provide a path')

    #run_experiment.load_gin_configs(FLAGS.gin_file)

    environment = run_experiment.create_environment()
    obs_stacker = run_experiment.create_obs_stacker(environment)
    agent = run_experiment.create_agent(environment, obs_stacker)

    run_experiment.run_experiment(agent, environment, obs_stacker)

def main():
    launch_experiment()

if __name__ == '__main__':
    main()
