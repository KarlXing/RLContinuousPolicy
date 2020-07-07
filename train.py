import torch
import torch.nn as nn
from torch.distributions import Beta, Normal
import torch.distributions.kl as kl

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog, ActionDistribution
from ray.rllib.utils.annotations import override

import argparse, tensorboard
import gym
import pybullet_envs


######## Args Setting ##########
parser = argparse.ArgumentParser()
parser.add_argument('--train-epochs', default=5000, type=int)
parser.add_argument('--trainer-save-path', default='./', type=str)
parser.add_argument('--model-save-freq', default=100, type=int)
parser.add_argument('--model-save-path', default='./', type=str)
parser.add_argument('--num-workers', default=5, type=int)
parser.add_argument('--num-envs-per-worker', default=2, type=int)
parser.add_argument('--num-gpus', default=1, type=int)
parser.add_argument('--use-gae', default=True, type=bool)
parser.add_argument('--batch-mode', default='truncate_episodes', type=str)
parser.add_argument('--vf-loss-coeff', default=1, type=int)
parser.add_argument('--vf-clip-param', default=10000, type=int)
parser.add_argument('--lr', default=0.00005, type=float)
parser.add_argument('--kl-coeff', default=0, type=float)
parser.add_argument('--num-sgd-iter', default=10, type=int)
parser.add_argument('--sgd-minibatch-size', default=128, type=int)
parser.add_argument('--grad-clip', default=0.5, type=float, help='other implementations may refer as max_grad_norm')
parser.add_argument('--rollout-fragment-length', default=1000, type=int)
parser.add_argument('--train-batch-size', default=10000, type=int)
parser.add_argument('--clip-param', default=0.1, type=float, help='other implementations may refer as clip_ratio')
parser.add_argument('--env-id', default='Walker2d-v2')
parser.add_argument('--dis', default='gaussian', type=str)
args = parser.parse_args()


class MyEnv(gym.Env):
    def __init__(self, env_config):
        import pybullet_envs
        self.env = gym.make(env_config['env_id'])
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        action = self.action_space.high * action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

register_env("myenv", lambda config: MyEnv(config))

class MyModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        custom_config = model_config['custom_options']
        self.dis = custom_config['dis']
        input_size, output_size = obs_space.shape[0], action_space.shape[0]

        self.actor = nn.Sequential(nn.Linear(input_size, 300), nn.ReLU(),
                                nn.Linear(300, 400), nn.ReLU())
        if custom_config['dis'] == 'gaussian':
            self.actor_h1 = nn.Sequential(nn.Linear(400, output_size), nn.Tanh())
            self.actor_h2 = nn.Sequential(nn.Linear(400, output_size), nn.Softplus())
        elif custom_config['dis'] == 'beta':
            self.actor_h1 = nn.Sequential(nn.Linear(400, output_size), nn.Softplus())
            self.actor_h2 = nn.Sequential(nn.Linear(400, output_size), nn.Softplus())
        else:
            print("Invalid Distribution Config")
            return

        self.critic = nn.Sequential(nn.Linear(input_size, 300), nn.ReLU(), nn.Linear(300, 400), nn.ReLU(), nn.Linear(400, 1))

        self._cur_value = None


    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs'].float()
        actor_features = self.actor(obs)

        if self.dis == 'gaussian':
            actor_h1 = self.actor_h1(actor_features)
            actor_h2 = self.actor_h2(actor_features)
        else:
            actor_h1 = self.actor_h1(actor_features)+1
            actor_h2 = self.actor_h2(actor_features)+1
        logits = torch.cat([actor_h1, actor_h2], dim=1)
        self._cur_value = self.critic(obs).squeeze(1)

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, 'Must call forward() first'
        return self._cur_value

ModelCatalog.register_custom_model("mymodel", MyModel)


class MyGaussianDist(ActionDistribution):
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return action_space.shape[0]*2

    def __init__(self, inputs, model):
        super(MyGaussianDist, self).__init__(inputs, model)
        mu, std = torch.split(inputs, int(inputs.shape[1]/2), dim=1)
        self.dist = Normal(mu, std)

    def sample(self):
        self.sampled_action = self.dist.sample()
        return self.sampled_action

    def deterministic_sample(self):
        return self.dist.mean

    def sampled_action_logp(self):
        result = self.logp(self.sampled_action)
        return result

    def logp(self, actions):
        return self.dist.log_prob(actions).sum(-1)

    def kl(self, other):
        p, q = self.dist, other.dist
        return kl.kl_divergence(p, q).sum(-1)

    def entropy(self):
        return self.dist.entropy().sum(-1)


class MyBetaDist(ActionDistribution):
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return action_space.shape[0]*2

    def __init__(self, inputs, model):
        super(MyBetaDist, self).__init__(inputs, model)
        alpha, beta = torch.split(inputs, int(inputs.shape[1]/2), dim=1)
        self.dist = Beta(alpha, beta)

    def sample(self):
        self.sampled_action = self.dist.sample()
        return self.sampled_action

    def deterministic_sample(self):
        return self.dist.mean

    def sampled_action_logp(self):
        return self.logp(self.sampled_action)

    def logp(self, actions):
        return self.dist.log_prob(actions).sum(-1)

    def kl(self, other):
        p, q = self.dist, other.dist
        return kl_divergence(p, q).sum(-1)

    def entropy(self):
        return self.dist.entropy().sum(-1)


ModelCatalog.register_custom_action_dist("mygaussiandist", MyGaussianDist)
ModelCatalog.register_custom_action_dist("mybetadist", MyBetaDist)


def main():
    ray.init()

    dis = "mygaussiandist" if args.dis == 'gaussian' else "mybetadist"
    
    #  Hyperparameters of PPO are not well tuned. Most of them refer to https://github.com/xtma/pytorch_car_caring/blob/master/train.py
    trainer = PPOTrainer(env="myenv", config={
        "use_pytorch": True,
        "model":{"custom_model":"mymodel",
                "custom_options":{"dis":args.dis},
                "custom_action_dist":dis,
                },
        "env_config":{'env_id':args.env_id},
        "num_workers":args.num_workers,
        "num_envs_per_worker":args.num_envs_per_worker,
        "num_gpus":args.num_gpus,
        "use_gae":args.use_gae,
        "batch_mode":args.batch_mode,
        "vf_loss_coeff":args.vf_loss_coeff,
        "vf_clip_param":args.vf_clip_param,
        "lr":args.lr,
        "kl_coeff":args.kl_coeff,
        "num_sgd_iter":args.num_sgd_iter,
        "grad_clip":args.grad_clip,
        "clip_param":args.clip_param,
        "rollout_fragment_length":args.rollout_fragment_length,
        "train_batch_size":args.train_batch_size,
        "sgd_minibatch_size":args.sgd_minibatch_size,
        })


    for i in range(args.train_epochs):
        trainer.train()
        print("%d Train Done" % (i), "Save Freq: %d" % (args.model_save_freq))
        if (i+1) % args.model_save_freq == 0:
            print("%d Episodes Done" % (i))
            weights = trainer.get_policy().get_weights()
            torch.save(weights, args.model_save_path+"%d-mode.pt" % (i+1))
    trainer.save(args.trainer_save_path)
    print("Done All!")
    trainer.stop()

if __name__ == '__main__':
    main()