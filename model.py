import torch
import torch.nn as nn
from torch.distributions import Beta, Normal
import torch.nn.functional as F


class BetaPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_layer=[64,64]):
        super(BetaPolicy, self).__init__()
        actor_layer_size = [input_dim] + hidden_layer
        actor_feature_layers = nn.ModuleList([])
        for i in range(len(actor_layer_size)-1):
        	actor_feature_layers.append(nn.Linear(actor_layer_size[i], actor_layer_size[i+1]))
        	actor_feature_layers.append(nn.ReLU())
        self.actor = nn.Sequential(*actor_feature_layers)
        self.alpha_head = nn.Sequential(nn.Linear(hidden_layer[-1], action_dim), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(hidden_layer[-1], action_dim), nn.Softplus())
    
        critic_layer_size = [input_dim] + hidden_layer
        critic_layers = nn.ModuleList([])
        for i in range(len(critic_layer_size)-1):
        	critic_layers.append(nn.Linear(critic_layer_size[i], critic_layer_size[i+1]))
        	critic_layers.append(nn.ReLU())
        critic_layers.append(nn.Linear(hidden_layer[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x, action=None):
        actor_features = self.actor(x)
        alpha = self.alpha_head(actor_features)+1
        beta = self.beta_head(actor_features)+1
        self.dist = Beta(alpha, beta)
        if action is None:
            action = self.dist.sample()
        else:
            action = (action+1)/2
        action_log_prob = self.dist.log_prob(action).sum(-1)
        entropy = self.dist.entropy().sum(-1)
        value = self.critic(x)

        return action*2-1, action_log_prob, value.squeeze(-1), entropy


class GaussianPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_layer=[64,64]):
        super(GaussianPolicy, self).__init__()
        actor_layer_size = [input_dim] + hidden_layer
        actor_feature_layers = nn.ModuleList([])
        for i in range(len(actor_layer_size)-1):
        	actor_feature_layers.append(nn.Linear(actor_layer_size[i], actor_layer_size[i+1]))
        	actor_feature_layers.append(nn.ReLU())
        self.actor = nn.Sequential(*actor_feature_layers)
        self.mu_head = nn.Sequential(nn.Linear(hidden_layer[-1], action_dim), nn.Tanh())
        self.std_head = nn.Sequential(nn.Linear(hidden_layer[-1], action_dim), nn.Softplus())
    
        critic_layer_size = [input_dim] + hidden_layer
        critic_layers = nn.ModuleList([])
        for i in range(len(critic_layer_size)-1):
        	critic_layers.append(nn.Linear(critic_layer_size[i], critic_layer_size[i+1]))
        	critic_layers.append(nn.ReLU())
        critic_layers.append(nn.Linear(hidden_layer[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x, action=None):
        actor_features = self.actor(x)
        mu = self.mu_head(actor_features)
        std = self.std_head(actor_features)
        self.dist = Normal(mu, std)
        if action is None:
            action = self.dist.sample()
        action_log_prob = self.dist.log_prob(action).sum(-1)
        entropy = self.dist.entropy().sum(-1)
        value = self.critic(x)

        return action, action_log_prob, value.squeeze(-1), entropy


class GaussianUnitStdPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_layer=[64,64]):
        super(GaussianUnitStdPolicy, self).__init__()
        actor_layer_size = [input_dim] + hidden_layer
        actor_feature_layers = nn.ModuleList([])
        for i in range(len(actor_layer_size)-1):
            actor_feature_layers.append(nn.Linear(actor_layer_size[i], actor_layer_size[i+1]))
            actor_feature_layers.append(nn.ReLU())
        self.actor = nn.Sequential(*actor_feature_layers)
        self.mu_head = nn.Sequential(nn.Linear(hidden_layer[-1], action_dim), nn.Tanh())
    
        critic_layer_size = [input_dim] + hidden_layer
        critic_layers = nn.ModuleList([])
        for i in range(len(critic_layer_size)-1):
            critic_layers.append(nn.Linear(critic_layer_size[i], critic_layer_size[i+1]))
            critic_layers.append(nn.ReLU())
        critic_layers.append(nn.Linear(hidden_layer[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x, action=None):
        actor_features = self.actor(x)
        mu = self.mu_head(actor_features)
        std = torch.ones_like(mu)
        self.dist = Normal(mu, std)
        if action is None:
            action = self.dist.sample()
        action_log_prob = self.dist.log_prob(action).sum(-1)
        entropy = self.dist.entropy().sum(-1)
        value = self.critic(x)

        return action, action_log_prob, value.squeeze(-1), entropy