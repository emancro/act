### Task parameters

import os

DATA_DIR = ''
TASK_CONFIGS = {
    'insert_redness_relief_slanted':{
        'dataset_dir': os.environ['DATA'] + '/closed_loop_demos/insert_redness_relief_slanted',
        'num_episodes': -1,
        'episode_len': 80,
        'camera_names': ['color_image'],
    },


    'insert_redness_relief_slanted_anywhere_middle':{
        'dataset_dir': [
                        os.environ['DATA'] + '/closed_loop_demos/insert_redness_relief_slanted_anywhere',
                        os.environ['DATA'] + '/closed_loop_demos/insert_redness_relief_slanted',
                        ],
        'num_episodes': -1,
        'episode_len': 80,
        'camera_names': ['color_image'],
    },

    'insert_redness_relief_slanted_anywhere':{
        'dataset_dir': [
                        os.environ['DATA'] + '/closed_loop_demos/insert_redness_relief_slanted_anywhere',
                        ],
        'num_episodes': -1,
        'episode_len': 80,
        'camera_names': ['color_image'],
    },
}

### ALOHA fixed constants
DT = 0.02

