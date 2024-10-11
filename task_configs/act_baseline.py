from .example_task import TASK_CONFIG, augment_images

TASK_CONFIG["train"]["validate_every"] = 10
TASK_CONFIG["train"]["dataset_dir"] = "data/original/Remove_lid_atm"