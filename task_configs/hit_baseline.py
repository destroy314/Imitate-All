from .act_baseline import TASK_CONFIG, augment_images

def policy_maker(config:dict, stage=None):
    from policies.act.hit import HITPolicy
    from policies.common.maker import post_init_policies
    policy = HITPolicy(config)
    post_init_policies([policy], stage, [config["ckpt_path"]])
    return policy


TASK_CONFIG["common"]["policy_config"]["policy_maker"] = policy_maker
TASK_CONFIG["common"]["policy_config"]["policy_class"] = "HIT"
# 183+chunk_size for 224,400 in HIT repo. don't know why yet
TASK_CONFIG["common"]["policy_config"]["context_len"] = 183 + TASK_CONFIG["common"]["policy_config"]["chunk_size"]
TASK_CONFIG["common"]["policy_config"]["use_pos_embd_image"] = True
TASK_CONFIG["common"]["policy_config"]["use_pos_embd_action"] = True
TASK_CONFIG["common"]["policy_config"]["self_attention"] = True
# future image feature reconstruction loss
TASK_CONFIG["common"]["policy_config"]["feature_loss"] = True
TASK_CONFIG["common"]["policy_config"]["feature_loss_weight"] = 0.005