import torch
import os
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import tempfile
from threading import Thread, Event

# Get the absolute path of the project root directory
parent_dir = os.path.dirname(os.path.realpath(__file__))
plate_detection_path = os.path.join(parent_dir,'PlateDetection')
area_segmentation_path = os.path.join(parent_dir,'FoodSegmentation')
image_search_path = os.path.join(parent_dir,'ImageSearch')

sys.path.append(parent_dir)
sys.path.append(plate_detection_path)
sys.path.append(area_segmentation_path)
sys.path.append(image_search_path)

# Define variables to store the return values
bboxes_result = None
embeddings_result = None

# Define events to signal when each thread has finished
bboxes_done_event = Event()
embeddings_done_event = Event()
packaged_bboxes_done_event = Event()

from FoodSegmentation.sam_model import GenerateMaskForImage,prepare_image_embeddings,prepare_image_embeddings_mobile_sam
from FoodSegmentation.utils import format_bbox,show_box_cv2,show_mask_cv2

from FoodDetection.food_detection import detect_food

from ImageSearch.find_neighbors import find_similar_images,init_matching_index_endpoint

from ultralytics.utils.plotting import Annotator, colors, save_one_box
import argparse
import csv
import platform
from pathlib import Path



# Add the project root directory to the Python path if not already present
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

if project_root not in sys.path:
    sys.path.append(project_root)

REGION = 'asia-east1'
PROJECT_NUMBER = 898528291092
INDEX_ENDPOINT_ID = '8255379591946829824'

DEPLOYED_INDEX_ID = "test_5_foods_1708362748643_1708976525434"
EMBEDDING_DIMENSION = 128


def get_food_masks(sam_predictor,
                   bboxes,
                   open=True,
                   close=True,
                   kernel_size=None):
    
    masks,iou = GenerateMaskForImage(sam_predictor, bounding_boxes=bboxes,open=open,close=close,kernel_size=kernel_size)
    
    return masks,iou

def calculate_surface_area(masks,
                           food_types):
    
    # Create a dictionary to store the sums of masks with the same name
    mask_dict = {}

    # Iterate over each mask and its corresponding name
    for mask, name in zip(masks, food_types):
        if name not in mask_dict:
            mask_dict[name] = mask*1
        else:
            mask_dict[name] += mask*1

    # Create a dictionary to store the count of ones in each array
    pixel_count = {}

    # Iterate over the dictionary items and sum in each mask
    for name, summed_mask in mask_dict.items():
        non_zero_count = np.sum(summed_mask)
        pixel_count[name] = non_zero_count.item()

    return pixel_count


def get_food_bboxes_worker(opt):
    global bboxes_result
    bboxes_result,_ = detect_food(image_path=opt["source"],model_path=opt["weights"])
    bboxes_done_event.set()

def prepare_image_embeddings_worker(image,model_type,mobile_sam):
    global embeddings_result
    if mobile_sam:
        embeddings_result = prepare_image_embeddings_mobile_sam(image,"vit_t")
    else:
        embeddings_result = prepare_image_embeddings(image,model_type)
    embeddings_done_event.set()

def crop(image, bbox):
    img_height, img_width, _ = image.shape
    x, y, w, h = map(int, bbox)  # Convert coordinates to integers
    cropped_image = image[y-int(h/2):y+int(h/2),x-int(w/2):x+int(w/2),:]
    return cropped_image

def save_cropped_image(cropped_image):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Generate a temporary file path
    temp_file_path = os.path.join(temp_dir, 'cropped_image.jpg')
    
    # Save the cropped image to the temporary file
    cv2.imwrite(temp_file_path, cropped_image)
    
    return temp_file_path





def pipeline(opt,index_endpoint=None):

    image = cv2.imread(opt["source"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Create two threads to run get_food_bboxes and prepare_image_embeddings concurrently

    get_bboxes_thread = Thread(target=get_food_bboxes_worker,args=(opt,))
    prepare_embeddings_thread = Thread(target=prepare_image_embeddings_worker, args=(image,
                                                                                     opt["segmentation_model_type"],
                                                                                     opt["mobile_sam"]))

    # Start the threads
    get_bboxes_thread.start()
    if opt["segment"]:
        prepare_embeddings_thread.start()


    # Wait for both threads to finish
    get_bboxes_thread.join()
    if opt["segment"]:
        prepare_embeddings_thread.join()


    # Now you can access the return values
    bboxes_done_event.wait()
    if opt["segment"]:
        embeddings_done_event.wait()

    # Access the return values
    bboxes = bboxes_result
    sam_predictor = embeddings_result
    
    if (len(bboxes) != 0) & opt["segment"]:
        masks,iou = get_food_masks(sam_predictor,
                            bboxes,
                            open=True,
                            close=True,
                            kernel_size=None)
    else : 
        masks = None
        iou = None

    
    food_types=[]
    if True:
        for bbox in bboxes:
            cropped_image = crop(image,bbox)
            crop_path = save_cropped_image(cropped_image)
            response = find_similar_images(image_path=crop_path,
                                        index_endpoint=index_endpoint,
                                        deployed_index_id=DEPLOYED_INDEX_ID)
            
            labels = [neighbor[0].split('\\')[0] for neighbor in response]
            most_common_label = max(set(labels), key=labels.count)
            food_types.append(most_common_label)

    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    #Image visualization
    if masks:
        for i,mask in enumerate(masks):
            image = show_mask_cv2(mask[0],image)

    for i,bbox in enumerate(bboxes):
        if masks:
            image = show_box_cv2(format_bbox(bbox), image,iou=iou[i][0],category_name=food_types[i])
        else:
            image = show_box_cv2(format_bbox(bbox), image,iou=None,category_name=food_types[i])

    if opt["save"]:
        if os.path.exists(r'./PipelineTestResults') == False:
            os.mkdir(r'./PipelineTestResults')
        cv2.imwrite(os.path.join('.','PipelineTestResults',f'test.jpg'), image)


    
    #Calculates masks pixel count and returns a dictionnary with surface area for every food {'food_type':pixel_count}
    if masks :
        pixel_count_dict = calculate_surface_area(
                                    masks,
                                    food_types)
    else:
        pixel_count_dict = {}
    
    bbox_dict = {}

    # Iterate over each bbox and its corresponding name
    for bbox, name in zip(bboxes, food_types):
        if name not in bbox_dict:
            bbox_dict[name] = list(bbox.numpy())
        else: 
            bbox_dict[name].append(list(bbox.numpy()))
    
    return pixel_count_dict,bbox_dict,image

    
if __name__ == '__main__':

    index_endpoint = init_matching_index_endpoint(project_number=PROJECT_NUMBER, region=REGION, index_id=INDEX_ENDPOINT_ID)

    start_time = time.time()
    
    # Define Arguments of Food Detection
    opt = {
        "weights": "./Models/best.pt",
        "mobile_sam": True,
        "segmentation_model_type": "vit_b",
        "source": r"test_images\2.jpg",
        "segment": True,
        "imgsz": (640, 640),
        "conf_thres": 0.01,
        "iou_thres": 0.7,
        "save": True
    }

    results = pipeline(opt,index_endpoint=index_endpoint)
    print(results)
    end_time = time.time()

    print('Elapsed time : ', end_time - start_time)
