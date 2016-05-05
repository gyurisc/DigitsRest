#!/usr/bin/env python2

"""
Classifies the model from ./model using the images from ./images folder
"""

import argparse
import os
import tarfile
import tempfile
import time
import zipfile

from classify_cli import classify
from classify_cli import print_classification_results

def classify_with_model_and_images_folder(model_folder, image_folder):
    """
    Classifying images from the i9mages folder using with the model from the model folder 
    """        
    caffemodel = None
    deploy_file = None
    mean_file = None
    labels_file = None
       
    # Model 
    for filename in os.listdir(model_folder):
        full_path = os.path.join(model_folder, filename)
        if filename.endswith('.caffemodel'):
            caffemodel = full_path
        elif filename == 'deploy.prototxt':
            deploy_file = full_path
        elif filename.endswith('.binaryproto'):
            mean_file = full_path
        elif filename == 'labels.txt':
            labels_file = full_path

    assert caffemodel is not None, 'Caffe model file not found'
    assert deploy_file is not None, 'Deploy file not found'
    
    print('Model OK!')
      
    # Images 
    image_files = [] 
    
    for image in os.listdir(image_folder):
        image_files.append(os.path.join(image_folder, image))

    assert image_files is not None, 'Test images are missing'
    assert len(image_files) != 0, 'Test images are missing'
    
    print('Images OK!')
    
    classifications = classify(caffemodel, deploy_file, image_files,
             mean_file=mean_file, labels_file=labels_file)
    
    print_classification_results(classifications, image_files)
            
if __name__ == '__main__':
    script_start_time = time.time()

    classify_with_model_and_images_folder('model', 'images')    
    print 'Script took %s seconds.' % (time.time() - script_start_time,)
