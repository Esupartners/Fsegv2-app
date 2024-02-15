from google.cloud import aiplatform
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from get_embeddings import encode_images_to_embeddings,encode_texts_to_embeddings

REGION = 'asia-east1'
PROJECT_NUMBER = 898528291092
INDEX_ENDPOINT_ID = '8255379591946829824'

DEPLOYED_INDEX_ID = "matching_test_512_1707923509842"

EMBEDDING_DIMENSION = 512




def display_similar_images_opencv(input_image_path, response):

    # Load the query image
    query_image = cv2.imread(input_image_path)

    # Display query image
    #cv2.imshow('Query Image', query_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Display similar images with proper text using OpenCV
    for i, neighbor in enumerate(response, 1):
        similar_image = cv2.imread(os.path.join(r'.\Test-2000',neighbor[0]))

        # Display similar image
        cv2.imshow(f'Similar Image {i}', cv2.resize(similar_image,(int(similar_image.shape[1]/2),int(similar_image.shape[0]/2))))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def get_matching_engine_response(index_endpoint, deployed_index_id, image_embeddings, num_neighbors=5):
    """
    Get matching engine response for similar images.

    Parameters:
    - index_endpoint (str): The endpoint of the matching engine index.
    - deployed_index_id (str): The ID of the deployed index.
    - image_embeddings (list): List of image embeddings for the query images.
    - num_neighbors (int): Number of neighbors to return.

    Returns:
    - dict: Matching engine response.
    """

    # Get embeddings using the matching engine index endpoint
    response = index_endpoint.find_neighbors(
        deployed_index_id=deployed_index_id,
        queries=image_embeddings,
        num_neighbors=num_neighbors,
    )

    return response

def init_matching_index_endpoint(project_number=PROJECT_NUMBER, region=REGION, index_id=INDEX_ENDPOINT_ID):
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(f'projects/{project_number}/locations/{region}/indexEndpoints/{index_id}')
    return index_endpoint

def find_similar_images(image_path=None, 
                        index_endpoint=None, 
                        deployed_index_id=DEPLOYED_INDEX_ID, 
                        num_neighbors=5, 
                        text=None):

    if text is None:
        embeddings = encode_images_to_embeddings(image_uris=[image_path],parameters={"dimension": EMBEDDING_DIMENSION})
    else:
        embeddings = encode_texts_to_embeddings(text=[text],parameters={"dimension": EMBEDDING_DIMENSION})

    response = get_matching_engine_response(index_endpoint, deployed_index_id, embeddings, num_neighbors=num_neighbors)

    return [(response[0][i].id,response[0][i].distance) for i in range(len(response[0]))]




if __name__ == "__main__":

    image_path = r'.\2.jpg'


    index_endpoint = init_matching_index_endpoint(project_number=PROJECT_NUMBER, region=REGION, index_id=INDEX_ENDPOINT_ID)
    response = find_similar_images(image_path=image_path,
                                   index_endpoint=index_endpoint,
                                   deployed_index_id=DEPLOYED_INDEX_ID)
    labels = [neighbor[0].split('\\')[0] for neighbor in response]
    print(labels)
    print(max(set(labels), key=labels.count))
    display_similar_images_opencv(image_path, response)






