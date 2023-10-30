from pfrl.agents import PPO
from .profile import PPOAgentProfile, AgentProfile
from pfrl.nn import Branched
from nets import common, mlp_policy
import torch


class AgentInitializer:

    @staticmethod
    def create_agent(profile, optimizer=None):
        """
        create agent using profile

        Args:
            profile (AgentProfile): AgentProfile that contains parameters to construct an agent

        """
        if profile.agent_type == "PPO":
            return AgentInitializer.create_ppo_agent(profile, optimizer)
        else:
            raise NotImplementedError("Unknown agent: %s" % profile.agent_type)

    @staticmethod
    def _load_policy(policy_type: str, profile: AgentProfile):
        if policy_type == "MLPWithSoftmaxHead":
            return mlp_policy.MLPWithSoftmaxHead(profile.obs_size, profile.policy_hiddens, profile.action_size)
        else:
            raise NotImplementedError("Unknown policy type: %s" % policy_type)

    @staticmethod
    def _load_activation(act_type):
        if act_type == "relu":
            return torch.relu
        elif act_type == "tanh":
            return torch.tanh
        else:
            return torch.relu

    @staticmethod
    def _load_vf(vf_type: str, profile: AgentProfile):
        if vf_type == "MLP":
            act = AgentInitializer._load_activation(profile.vf_act)
            return common.MLP([profile.obs_size] + profile.vf_hiddens + [1], activation=act)
        else:
            raise NotImplementedError("Unknonwn value type: %s" % vf_type)

    @staticmethod
    def _load_model(profile: AgentProfile):
        if profile.agent_type == "PPO":
            policy = AgentInitializer._load_policy(profile.policy_type, profile)
            vf = AgentInitializer._load_vf(profile.vf_type, profile)
            model = Branched(policy, vf)
        else:
            raise NotImplementedError("Unknown agent type: %s"%profile.agent_type)
        
        return model

    @staticmethod
    def create_ppo_agent(ppo_profile: PPOAgentProfile, optimizer=None):
        model = AgentInitializer._load_model(ppo_profile)
        agent = PPO(model, optimizer=optimizer)
        return agent
