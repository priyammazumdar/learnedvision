import os
import random
from tqdm import tqdm
import shutil

def load_data(IMG_DIR):
    data = []
    counter = 0
    idx = 0
    for SPECIES in tqdm(os.listdir(IMG_DIR)):
        SPECIES_DIR = f"{IMG_DIR}/{SPECIES}"
        for img in os.listdir(SPECIES_DIR):
            rand = random.random()
            ORI_DIRECTORY = f"{SPECIES_DIR}/{img}"
            TEST_DIRECTORY = f"test/{SPECIES}/{img}"
            TRAIN_DIRECTORY = f"train/{SPECIES}/{img}"

            if rand < 0.1:
                if not os.path.exists(f'/test/{SPECIES}'):
                    os.makedirs(f'/test/{SPECIES}')
                shutil.copyfile(ORI_DIRECTORY, TEST_DIRECTORY)
            else:
                if not os.path.exists(f'/train/{SPECIES}'):
                    os.makedirs(f'/test/{SPECIES}')
                shutil.copyfile(ORI_DIRECTORY, TRAIN_DIRECTORY)


load_data('data')
