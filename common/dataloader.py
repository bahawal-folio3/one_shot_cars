from load_model import get_reid_embeddings
import os
import cv2
import numpy as np
import random
import itertools 
import pickle
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="This is a simple argument parser template.")
parser.add_argument("data_source", help="Path to the data contain car images")
parser.add_argument("data_save", help="Path where generated embedding will be saved")
parser.add_argument("split", help="specify whether it's a train/test/val split")

args = parser.parse_args()

def get_cars(base_dir):
    return [x for x in os.listdir(base_dir) if (os.path.isdir(f"{base_dir}/{x}" )) ]

def main():

    base_dir = args.data_source
    mode = args.split

    imgs = []
    for source_dir in get_cars(base_dir):
        source_dir_path = f'{base_dir}/{source_dir}'
        my_list = os.listdir(source_dir_path)
        imgs.extend([source_dir +'-'+ x for x in my_list])


    combinations = list(itertools.combinations(imgs, 2))
    len(combinations)

    same_pair = []
    diff_pair = []
    for pair in combinations:
        if pair[0].split('-')[0] == pair[1].split('-')[0]:
            same_pair.append(pair)
        else:
            diff_pair.append(pair)

    print(f"lenght of different pairs :", len(diff_pair))
    print(f"lenght of same pairs:", len(same_pair))

    chopped_diff_pairs = random.sample(diff_pair, int( len(same_pair)*3))

    print(f"length of ", len(chopped_diff_pairs))
    new_combined_list = same_pair + chopped_diff_pairs
    print(len(new_combined_list))
    new_combined_list[:5]
    if os.path.exists('data_set.pkl'):
        with open("data_set.pkl", "rb") as f:
            # Load the data from the file
            new_combined_list = pickle.load(f)
        # assert "stop"
    else:
        with open("data_set.pkl", "wb") as f:
            pickle.dump(new_combined_list, f)

    print(len(new_combined_list))
    new_combined_list[:5]

    unique_images = set()
    for a,b in new_combined_list:
        unique_images.add(a)
        unique_images.add(b)
    len(unique_images)

    unique_images = list(unique_images)
    unique_images[0]


    embedding_map = {}
    for img in tqdm(unique_images):
        img_dir,img_name = img.split('-')[0], '-'.join(img.split('-')[1:]) 
        image = cv2.imread(f"{base_dir}/{img_dir}/{img_name}")

        try:
            embedding_map[img] = get_reid_embeddings(cv2.resize(image,(256,256))/255)
        except:
            print(f'count not read {base_dir}/{img_dir}/{img_name}')

    x = []
    y = []
    count = 1
    target_dir = args.data_save
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    for pair in tqdm(new_combined_list):
        img1 = pair[0] 
        img2 = pair[1]
        if img1.endswith('.jpg')  and img2.endswith('jpg'):
            img1_label = pair[0].split('-')[0]
            img2_label = pair[1].split('-')[0]
            img1_embedding = embedding_map[img1]
            img2_embedding = embedding_map[img2]
            embedding = np.concatenate((img1_embedding, img2_embedding), axis=0)
            label = img2_label==img1_label
            np.save(f'{target_dir}/{count}.npy',np.array([embedding,label]))
            count+=1

if __name__ == "__main__":
    main()