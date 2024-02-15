import json
from tqdm import tqdm
import glob
import tempfile
import os

from get_embeddings_batched import encode_to_embeddings_chunked
from get_embeddings import encode_images_to_embeddings

# Create temporary file to write embeddings to
embeddings_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)

image_paths = glob.glob('Test-2000\*\*')
print(image_paths[:5])
image_names = [name.replace('Test-2000\\','') for name in image_paths]


BATCH_SIZE = 16
EMBEDDINGS_DIMENSIONS = 512

with open(embeddings_file.name, "a") as f:
    for i in tqdm(range(0, len(image_names), BATCH_SIZE)):
        image_names_chunk = image_names[i : i + BATCH_SIZE]
        image_paths_chunk = image_paths[i : i + BATCH_SIZE]

        embeddings = encode_to_embeddings_chunked(
            process_function=encode_images_to_embeddings, items=image_paths_chunk, embedding_dimension= EMBEDDINGS_DIMENSIONS
        )
        # Append to file
        embeddings_formatted = [
            json.dumps(
                {
                    "id": str(id),
                    "embedding": [str(value) for value in embedding],
                }
            )
            + "\n"
            for id, embedding in zip(image_names_chunk, embeddings)
            if embedding is not None
        ]
        f.writelines(embeddings_formatted)

BUCKET_URI = 'gs://image-search-test-2024'
UNIQUE_FOLDER_NAME = "embeddings_2000_512"
EMBEDDINGS_INITIAL_URI = f"{BUCKET_URI}/{UNIQUE_FOLDER_NAME}/"

os.system(f'gsutil cp {embeddings_file.name} {EMBEDDINGS_INITIAL_URI}')
