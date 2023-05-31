from load_model import get_reid_embeddings
import os
import cv2
import numpy as np
import random

def get_cars():
    return [x for x in os.listdir('data') if (os.path.isdir(f"data/{x}")) ]


imgs = []
sample_size = 5
for source_dir in get_cars():
    source_dir_path = f'data/{source_dir}'
    my_list = os.listdir(source_dir_path)
    random_samples = random.sample(my_list, min(sample_size, len(my_list)))
    imgs.extend([source_dir +'-'+ x for x in random_samples])
                         
with open('dataset_list.txt', 'w') as fp:
    for line in imgs:
        fp.write(line+'\n')

from itertools import combinations
combinations = list(combinations(imgs, 2))

from tqdm import tqdm
x = []
y = []
count = 1
for pair in tqdm(combinations):
    img1 = cv2.imread(f"data/{pair[0].split('-')[0]}/{pair[0].split('-')[1]}")
    img2 = cv2.imread(f"data/{pair[1].split('-')[0]}/{pair[1].split('-')[1]}")
#     print(f"data/{pair[0].split('-')[0]}/{pair[0].split('-')[1]}")
    img1_embedding = get_reid_embeddings(cv2.resize(img1,(256,256))/255)
    img2_embedding = get_reid_embeddings(cv2.resize(img2,(256,256))/255)
    embedding = np.concatenate((img1_embedding, img2_embedding), axis=0)
    y.append(int(pair[1].split('-')[0]==pair[0].split('-')[0]))
    x.append(embedding)
    if count % 10000 == 0:
        x_data = np.array(x)
        np.save(f'data/car_matching_dataset-{count}.npy', x_data)
        y_data = np.array(y)
        np.save(f'data/car_matching_label-{count}.npy', y_data)
        x = []
        y = []
    count+=1
