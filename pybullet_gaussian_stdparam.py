import rlcodebase
from rlcodebase.env import make_vec_envs
from rlcodebase.agent import PPOAgent
from rlcodebase.utils import get_action_dim, init_parser, Config, Logger
from torch.utils.tensorboard import SummaryWriter
from model import GaussianStdParamPolicy
import pybullet_envs
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--game', default='HalfCheetahBulletEnv-v0')
parser.add_argument('--seed', default=1, type=int)
args = parser.parse_args()

def main():
    # create config
    config = Config()
    config.game = args.game
    config.algo = 'ppo'
    config.max_steps = int(2e6)
    config.num_envs = 1
    config.optimizer = 'RMSprop'
    config.lr = 0.0003
    config.discount = 0.99
    config.use_gae = True
    config.gae_lambda = 0.95
    config.use_grad_clip = True
    config.max_grad_norm = 0.5
    config.rollout_length = 2048
    config.value_loss_coef = 0.5
    config.entropy_coef = 0
    config.ppo_epoch = 10
    config.ppo_clip_param = 0.2
    config.num_mini_batch = 32
    config.use_gpu = True
    config.seed = args.seed
    config.num_frame_stack = 1
    config.after_set()
    print(config)

    # prepare env, model and logger
    env = make_vec_envs(config.game, num_envs = config.num_envs, seed = config.seed, num_frame_stack= config.num_frame_stack)
    model = GaussianStdParamPolicy(env.observation_space.shape[0], action_dim = get_action_dim(env.action_space)).to(config.device)
    logger =  Logger(SummaryWriter(config.save_path), config.num_echo_episodes)

    # create agent and run
    agent = PPOAgent(config, env, model, logger)
    agent.run()

if __name__ == '__main__':
    main()
