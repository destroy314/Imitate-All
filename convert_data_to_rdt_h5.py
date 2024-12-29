import os
import time
import h5py
import json
import cv2
from tqdm import tqdm

def save_data(low_dim, image_paths, write_path):
    camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']

    data_size = len(low_dim['action/arm/joint_position'])
    data_dict = {
        '/observations/qpos': [],
        '/action': [],
    }

    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    qpos = low_dim['observation/arm/joint_position']
    actions = low_dim['action/arm/joint_position']
    gripper_pos = low_dim['observation/eef/joint_position']
    gripper_actions = low_dim['action/eef/joint_position']

    for i in range(data_size):
        data_dict['/observations/qpos'].append(qpos[i][:6]+gripper_pos[i][0:1]+qpos[i][6:]+gripper_pos[i][1:2])

        data_dict['/action'].append(actions[i][:6]+gripper_actions[i][0:1]+actions[i][6:]+gripper_actions[i][1:2])

        for cam_name, img_path in zip(camera_names, image_paths):
            img = cv2.imread(f"{img_path}/frame_{i:06}.png")
            cv2.imencode()
            data_dict[f'/observations/images/{cam_name}'].append(img)
    
    with h5py.File(write_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            _ = image.create_dataset(cam_name, (data_size, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )

        _ = obs.create_dataset('qpos', (data_size, 14))
        _ = root.create_dataset('action', (data_size, 14))

        for name, array in data_dict.items():  
            root[name][...] = array

def main():
    raw_dir = "data/raw"
    name = "transfer_block"
    dataset_dir = "data/dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    raw_data_dir = f"{raw_dir}/{name}/"
    episodes_dir = os.listdir(raw_data_dir)
    episodes_dir = sorted([x for x in episodes_dir if x.isdigit()])
    for episode in tqdm(episodes_dir):
        episode_dir = f"{raw_data_dir}/{episode}"
        imgs_dirs = os.listdir(episode_dir)
        imgs_dirs = sorted([f"{episode_dir}/{x}" for x in imgs_dirs if os.path.isdir(f"{episode_dir}/{x}")])
        with open(f"{episode_dir}/low_dim.json", 'r') as f:
            low_dim = json.load(f)
        save_data(low_dim, imgs_dirs, f"{dataset_dir}/{name}/{episode}")
    
    # instruction_dict = {
    #     'instruction': "Pick up the red block and place it on table with other arm.",
    #     'simplified_instruction': ["Pick up the block and put it on the other side of table."],
    #     'expanded_instruction': ["Pick up the red block with the closest arm, transfer to another arm, and place it on table."],
    # }
    instruction_dict = {
        'instruction': "Put the block into the cardboard box.",
        'simplified_instruction': ["Place the block into the box."],
        'expanded_instruction': ["Pick up the block carefully, carry it to the cardboard box, and place it inside the box."],
    }
    with open(f"{dataset_dir}/{name}/expanded_instruction_gpt-4-turbo.json", 'w') as f:
        json.dump(instruction_dict, f)

if __name__ == "__main__":
    main()
