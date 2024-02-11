from ultralytics import YOLO


MODEL_PATH = r'best.pt'

PREDICT_ARGS = {
    # Detection Settings
    'conf': 0.2,  # object confidence threshold for detection
    'iou': 0.3,  # intersection over union (IoU) threshold for NMS
    'imgsz': 640,  # image size as scalar or (h, w) list, i.e. (640, 480)
    'half': False,  # use half precision (FP16)
    'device': None,  # device to run on, i.e. cuda device=0/1/2/3 or device=cpu
    'max_det': 300,  # maximum number of detections per image
    'vid_stride': False,  # video frame-rate stride
    'stream_buffer': False,  # buffer all streaming frames (True) or return the most recent frame (False)
    'visualize': False,  # visualize model features
    'augment': False,  # apply image augmentation to prediction sources
    'agnostic_nms': False,  # class-agnostic NMS
    'classes': None,  # filter results by class, i.e. classes=0, or classes=[0,2,3]
    'retina_masks': False,  # use high-resolution segmentation masks
    'embed': None,  # return feature vectors/embeddings from given layers

    # Visualization Settings
    'show': False,  # show predicted images and videos if environment allows
    'save': False,  # save predicted images and videos
    'save_frames': False,  # save predicted individual video frames
    'save_txt': False,  # save results as .txt file
    'save_conf': False,  # save results with confidence scores
    'save_crop': False,  # save cropped images with results
    'show_labels': True,  # show prediction labels, i.e. 'person'
    'show_conf': True,  # show prediction confidence, i.e. '0.99'
    'show_boxes': True,  # show prediction boxes
    'line_width': None  # line width of the bounding boxes. Scaled to image size if None.
    }

def load_model(model_path='yolov8.pt'):

    # Load a model
    model = YOLO(model=model_path,task='detect')

    return model


def detect_food(image_path=None,model_path=MODEL_PATH):

    model = load_model(MODEL_PATH)

    # Run inference
    results = model.predict(source=r'.jpg',**PREDICT_ARGS)

    bounding_boxes = results.boxes.xywh
    scores = results.boxes.conf

    return bounding_boxes, scores
