import json

from utils.general import pprint_summary


class AgentProfile:
    """
    AgentProfile hold the basic parameters that defines an (PFRL) Agent.
    Note the parameter specifically for training/evalutaion will note be defined in this profile.
    """

    def __init__(self, ) -> None:
        self._required_fields = {}
        self._choices = {}
        self.agent_type = ""
        self._update_required_fields({
            "obs_size": int,
            "action_size": int
        })

    def _update_required_fields(self, fields):
        self._required_fields.update(fields)

    def _validate_fields(self, kv_profile):
        for key in self._required_fields:
            if key not in kv_profile:
                raise RuntimeError("required field `%s` not provided" % key)

            # validate type
            if type(kv_profile[key]) != self._required_fields[key]:
                raise RuntimeError("expect field `%s` have type=%s, while type=%s" % (
                    key, self._required_fields[key], type(kv_profile[key])))

            # validate choice
            if key in self._choices and kv_profile[key] not in self._choices[key]:
                raise RuntimeError("Unknown value %s for key: `%s`, require value in %s" % (
                    kv_profile[key], key, self._choices[key]))

    def load(self, filepath, agent_profile=None):
        with open(filepath) as f:
            dprofile = json.load(f)

        return self.read_from_dict(dprofile, agent_profile=agent_profile)

    def save(self, filepath):
        params = {k: v for k, v in self.__dict__.items()
                  if not k.startswith("_")}
        with open(filepath, "w+") as f:
            json.dump(params, f, sort_keys=True, )

    def read_from_dict(self, kv_profile, agent_profile=None):
        self._validate_fields(kv_profile)
        if not agent_profile:
            agent_profile = self

        agent_profile.__dict__.update(kv_profile)
        return agent_profile

    def add_choice(self, key, choice: list):
        self._choices[key] = choice

    def pprint(self, file=None):
        classname = self.__class__.__name__
        params = {k: v for k, v in self.__dict__.items()
                  if not k.startswith("_")}
        pprint_summary(classname, file, **params)


class PPOAgentProfile(AgentProfile):
    def __init__(self) -> None:
        super().__init__()
        self.agent_type = "PPO"
        self._update_required_fields({
            "policy_type": str,
            "policy_hiddens": list,
            "vf_type": str,
            "vf_hiddens": list,
            "vf_act": str,
        })
        self.add_choice("policy_type", ["MLPWithSoftmaxHead"])
        self.add_choice("vf_type", ["MLP"])
        
