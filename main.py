
    
        
     import os
import struct
import sys
import imageio
import numpy as np
from sklearn.cluster import KMeans
import torch
import tensorflow as tf
from scipy import ndimage
import cv2

def print_greeting():
    print("""
    # usage: process_icons <dir_path> <cluster_size1> ... <cluster_sizeN>
    # This script reads ICO files, extracts information about the contained images,
    # performs k-means clustering on pixel data, and saves the clustered images.
    #
    # example: process_icons.py icons_folder 5 10
    #   will create folders for each specified number of clusters with subfolders 0, 2, ..., n-1
    #   and save the corresponding clustered images inside those folders.
    """)

def read_images(images_path):
    images = []
    names = []
    for name in os.listdir(images_path):
        full_path = os.path.join(images_path, name)
        
        # Load the image using OpenCV
        image = cv2.imread(full_path)
        
        # Convert image to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Flatten the image and add it to the list
        images.append(image_gray.flatten())
        names.append(full_path)
    return images, names

def perform_clustering(images, n_clusters):
    #
    images_tensor = torch.tensor(images, dtype=torch.float32)
    
    # Run k-means clustering using sklearn
    kmeans_cluster = KMeans(n_clusters=n_clusters)
    kmeans_cluster.fit(images_tensor.numpy())
    
    
    labels = kmeans_cluster.labels_
    return labels

def create_clusters_folders(images_path, n_clusters, labels, names):
    for n in n_clusters:
        for cluster in range(n):
            folder = os.path.join(images_path, '..', f'{n}_clusters', str(cluster))
            os.makedirs(folder, exist_ok=True)
        
        for i, label in enumerate(labels):
            cluster = label % n
            item = os.path.join(images_path, '..', f'{n}_clusters', str(cluster), os.path.basename(names[i]))
            shutil.copy(names[i], item)

def main():
    if len(sys.argv) == 1:
        print_greeting()
        sys.exit(0)

    images_path = sys.argv[1]
    n_clusters = list(map(int, sys.argv[2:]))
    
   
    images, names = read_images(images_path)

    for n in n_clusters:
        print(f'Computing for {n} clusters...')
        
     
        labels = perform_clustering(images, n)
        

        create_clusters_folders(images_path, n, labels, names)

if __name__ == '__main__':
    main()
