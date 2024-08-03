from task_configs.template import (
    get_task_name,
    replace_task_name,
    set_paths,
    activator,
    TASK_CONFIG_DEFAULT,
)

def policy_maker(config:dict):
    from policy import ACTPolicy
    return ACTPolicy(config)

@activator(False)
def augment_images(image):
    from task_configs.config_augmentation.image.basic import color_transforms_1
    return color_transforms_1(image)

# auto replace the task name in the default paths accoring to the file name
TASK_NAME = get_task_name(__file__)
replace_task_name(TASK_NAME, stats_name="dataset_stats.pkl", time_stamp="now")
# but we also show how to set the whole paths manually
# DATA_DIR = f"./data/hdf5/{TASK_NAME}"
# CKPT_DIR = f"./my_ckpt/{TASK_NAME}/ckpt"
# STATS_PATH = f"./my_ckpt/{TASK_NAME}/dataset_stats.pkl"
# set_paths(DATA_DIR, CKPT_DIR, STATS_PATH)  # replace the default data and ckpt paths

chunk_size = 25
joint_num = 7
TASK_CONFIG_DEFAULT["common"]["camera_names"] = ["0"]
TASK_CONFIG_DEFAULT["common"]["state_dim"] = joint_num
TASK_CONFIG_DEFAULT["common"]["action_dim"] = joint_num
TASK_CONFIG_DEFAULT["common"]["policy_config"]["temporal_agg"] = False
TASK_CONFIG_DEFAULT["common"]["policy_config"]["chunk_size"] = chunk_size
TASK_CONFIG_DEFAULT["common"]["policy_config"]["num_queries"] = chunk_size
TASK_CONFIG_DEFAULT["common"]["policy_config"]["kl_weight"] = 10
TASK_CONFIG_DEFAULT["common"]["policy_config"]["policy_maker"] = policy_maker

TASK_CONFIG_DEFAULT["train"]["num_episodes"] = "ALL"
TASK_CONFIG_DEFAULT["train"]["num_epochs"] = 500
TASK_CONFIG_DEFAULT["train"]["batch_size"] = 4
TASK_CONFIG_DEFAULT["train"]["learning_rate"] = 2e-5
TASK_CONFIG_DEFAULT["train"]["pretrain_ckpt_path"] = ""
TASK_CONFIG_DEFAULT["train"]["pretrain_epoch_base"] = "AUTO"

TASK_CONFIG_DEFAULT["eval"]["robot_num"] = 1
TASK_CONFIG_DEFAULT["eval"]["joint_num"] = joint_num
TASK_CONFIG_DEFAULT["eval"]["start_joint"] = "AUTO"
TASK_CONFIG_DEFAULT["eval"]["max_timesteps"] = 300
TASK_CONFIG_DEFAULT["eval"]["ensemble"] = None

# final config
TASK_CONFIG = TASK_CONFIG_DEFAULT