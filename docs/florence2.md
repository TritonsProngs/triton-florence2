# Florence-2
This is an ensemble deployment of the [Florence-2-large-ft](https://huggingface.co/mrhendrey/Florence-2-large-ft-safetensors) model. This is a vision model that has
been finetuned for the following task prompts:

* "<CAPTION>"
* "<DETAILED_CAPTION>"
* "<MORE_DETAILED_CAPTION>"
* "<CAPTION_TO_PHRASE_GROUNDING>"
  * This takes an additional text prompt to do the grounding, e.g., "A cyclist holding a sign"
* "<OD>" (Object Detection)
* "<DENSE_REGION_CAPTION>"
* "<REGION_PROPOSAL>"
* "<OCR>"
* "<OCR_WITH_REGION>"

The output returned will be a JSON string that depends upon the task prompt given.

## Example Request
An image is submitted as a base64 encoded string.

```
import base64
import json
import requests
from uuid import uuid4

base_url = "http://localhost:8000/v2/models"

image_path = "/path/to/your/image.png"
with open(image_path, "rb") as f:
    image_bytes = f.read()
image_b64 = base64.b64encode(image_bytes).decode("UTF-8")

task_prompt = "<CAPTION>"

inference_request = {
    "id": uuid4().hex,
    "inputs": [
        {
            "name": "INPUT_IMAGE",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [image_b64],
        },
        {
            "name": "TASK_PROMPT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [task_prompt],
        }
    ]
}
response = requests.post(
    url=f"{base_url}/florence2/infer",
    json=inference_request,
).json()
result = json.loads(response["outputs"][0]["data"][0])
print(result)
# {'<CAPTION>': 'A woman holds up a sign that reads "safe streets now."'}
```

## Sending Many Images
If you need to send multiple images, it's important that you send each image request
separately and using multiple threads to ensure optimal throughput.

NOTE: You will encounter an OSError "Too many open files" if you send a lot of
requests. Typically the default ulimit is 1024 on most systems. Either increase this
using `ulimit -n {n_files}`, or don't create too many futures before you process them
as they are completed.

```
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import json
from pathlib import Path
import requests
from uuid import uuid4

def inference_request(image_path, task_prompt, text_prompt=None):
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("UTF-8")
    request = {
        "id": uuid4().hex,
        "inputs": [
            {
                "name": "INPUT_IMAGE",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [image_b64],
            },
            {
                "name": "TASK_PROMPT",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [task_prompt],
            }
        ]
    }
    if text_prompt:
        request["inputs"].append(
            {
                "name": "TEXT_PROMPT",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [text_prompt],
            }
        )
    return request

base_url = "http://localhost:8000/v2/models"

task_prompt = "<CAPTION>"
input_dir = Path("/path/to/image/directory/")
futures = {}
results = {}
with ThreadPoolExecutor(max_workers=4) as executor:
    for path in input_dir.iterdir():
        if path.is_file():
            future = executor.submit(
                requests.post,
                url=f"{base_url}/florence2/infer",
                json=inference_request(path, task_prompt, text_prompt=None),
            )
            futures[future] = str(path.absolute())
    
    for future in as_completed(futures):
        try:
            response = future.result().json()
        except Exception as exc:
            print(f"{futures[future]} threw {exc}")
        else:
            path = futures[future]
            try:
                parsed_json = json.loads(response["outputs"][0]["data"][0])
                results[path] = parsed_json
            except Exception as exc:
                print(f"{path} failed to get response: {exc}")

```

## Performance Analysis