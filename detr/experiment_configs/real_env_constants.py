### Task parameters

import os

DATA_DIR = ''
TASK_CONFIGS = {
    # this config is only inserting in the middle
    'insert_redness_relief_slanted':{
        'dataset_dir': os.environ['DATA'] + '/closed_loop_demos/insert_redness_relief_slanted',
        'num_episodes': -1,
        'episode_len': 80,
        'camera_names': ['color_image'],
    },

    # this config is only inserting anywhere
    'insert_redness_relief_slanted_anywhere':{
        'dataset_dir': [
                        os.environ['DATA'] + '/closed_loop_demos/insert_redness_relief_slanted_anywhere',
                        ],
        'num_episodes': -1,
        'episode_len': 80,
        'camera_names': ['color_image'],
    },

    # this config is only inserting anywhere
    'pick_vial':{
        'dataset_dir': [
                        os.environ['DATA'] + '/closed_loop_demos/pick_place_phenylephrine/',
                        ],
        'stage_key': '1_pick',
        'num_episodes': -1,
        'episode_len': 80,
        'camera_names': ['color_image'],
    },

    'place_vial':{
        'dataset_dir': [
                        os.environ['DATA'] + '/closed_loop_demos/pick_place_phenylephrine/',
                        ],
        'stage_key': '2_place',
        'num_episodes': -1,
        'episode_len': 80,
        'camera_names': ['color_image'],
    },
}

### ALOHA fixed constants
DT = 0.02

