from fastapi import FastAPI, File, UploadFile,Query
from fastapi.responses import FileResponse
from PIL import Image
import os
import json
import base64
import cv2
from main import pipeline


app = FastAPI()

# Directory to save temporary uploaded files
upload_dir = "uploads"
transformed_image_dir = "transformed_image"

# Create the directory if it doesn't exist
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
    
if not os.path.exists(transformed_image_dir):
    os.makedirs(transformed_image_dir)


@app.post("/uploadfile/")
async def upload_file(file: UploadFile):
        # Define Arguments of Food Detection
    OPT = {
        "weights": "./Models/best.pt",
        "segmentation_model_type": "vit_b",
        "source": os.path.join(upload_dir, file.filename),
        "segment": True,
        "imgsz": (640, 640),
        "conf_thres": 0.25,
        "iou_thres": 0.45,
        "save": True
    }


    # Save the uploaded file to a temporary location
    with open(os.path.join(upload_dir, file.filename), "wb") as f:
        f.write(file.file.read())

    # Get the MIME type of the image
    extension = os.path.splitext(file.filename)[1].removeprefix('.')
    mime_type = f"image/{extension}" 

    pixel_count_dict,bbox_dict,transformed_image = pipeline(OPT)

    # Save the transformed image to a temporary location
    transformed_image_path = os.path.join(transformed_image_dir, "transformed_" + file.filename)
    cv2.imwrite(transformed_image_path, transformed_image)

    # Encode the transformed image as Base64
    #with open(transformed_image_path, "rb") as img_file:
        #transformed_image_data = base64.b64encode(img_file.read()).decode("utf-8")

    transformed_image_url = f"/get_transformed_image/{file.filename}"

    # Create a response JSON that includes the Base64-encoded image, pixel_count, and MIME type
    response_data = {
        "transformed_image": transformed_image_url,
        "plate_food" : {
            "bounding_boxes": bbox_dict,
            "pixel_count": pixel_count_dict
        },            
        "mime_type": mime_type
    }

    return response_data

@app.get("/get_transformed_image/{filename}")
async def get_transformed_image(
    filename: str,
    content_type: str = Query("image/jpeg", title="Content-Type Header")
):
    transformed_image_path = os.path.join(transformed_image_dir, "transformed_" + filename)
    if os.path.exists(transformed_image_path):
        return FileResponse(transformed_image_path, headers={"Content-Type": content_type})
    else:
        return {"error": "Image not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
