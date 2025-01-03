# Florence-2
This is an ensemble deployment of the [Florence-2-large-ft](https://huggingface.co/mrhendrey/Florence-2-large-ft-safetensors) model. This is a vision model that has
been finetuned for the following task prompts:

* "\<CAPTION>"
* "<DETAILED_CAPTION>"
* "<MORE_DETAILED_CAPTION>"
* "<CAPTION_TO_PHRASE_GROUNDING>"
  * This takes an additional text prompt to do the grounding, e.g., "A cyclist holding a sign"
* "\<OD>" (Object Detection)
* "<DENSE_REGION_CAPTION>"
* "<REGION_PROPOSAL>"
* "\<OCR>"
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
The throughput performance is highly task dependent with some tasks taking much longer
than others. The ones that have regions (OD, OCR_WITH_REGION, etc.) can take
significantly longer depending on the number of objects in the image.

I use the following image during performance testing. As you can see, it has a lot of objects which will greatly affect throughput on tasks that get regions.
![Bike Protest](../data/July_30_Bike_Protest_crowd_-_safe_streets_now.jpg)

There is some sample data in [data/](../data/) directory. 

Launch the Triton Inference Server SDK container 

```sh
docker run \
  --rm -it --net host \
  -v ./:/workspace/triton-florence2 \
  nvcr.io/nvidia/tritonserver:24.10-py3-sdk
```

Inside the container, run the perf_analyzer CLI

```sh
sdk-container:/workspace perf_analyzer \
    -m florence2 \
    -v \
    --measurement-mode time_windows \
    --measurement-interval 20000 \
    --concurrency-range 30 \
    --input-data triton-florence2/data/load_data_CAPTION.json
```

| Task | Concurrency | Throughput (infer/s) | Ave. Latency (s) |
| ---- | ----------- | -------------------- | ------------ |
| \<CAPTION> | 30 | 25.9 | 1.15 |
| \<DETAILED_CAPTION> | 30 | 13.8 | 2.13 |
| \<MORE_DETAILED_CAPTION> | 30 | 19.2 | 1.59 |
| \<CAPTION_TO_PHRASE_GROUNDING> | 30 | 29.7 | 1.01 |
| \<OD> | 20 | 5.0 | 3.84 |
| \<DENSE_REGION_CAPTION> | 30 | 7.5 | 4.02 |
| \<REGION_PROPOSAL> | 30 | 6.3 | 4.72 |
| \<OCR> | 30 | 29.3 | 1.02 |
| \<OCR_WITH_REGION> | 30 | 21.3 | 1.39 |

## Validation
To validate this implementatin of the Florence-2 model, we calculate a few metrics that
correspond with the CAPTION task and compare the results against those published in the
[paper](https://arxiv.org/abs/2311.06242) and also against some of the metrics reported
in the [MS COCO paper](https://arxiv.org/pdf/1504.00325) paper for human performance
(See Table 1) for the [MS Coco 2014 Validation Data](https://cocodataset.org/#download).

This validation code is not quite as nicely contained as I would have liked so I'll
describe the process.

### Download Data
From the [COCO](https://cocodataset.org/#download) you can download the validation
images:

* [2014 Val images](http://images.cocodataset.org/zips/val2014.zip) (6GB)

### Clone coco-caption Repository
Clone the forked [COCO-Caption Repository](https://github.com/mtanti/coco-caption)
which has been updated for Python 3. This repo makes it easy and consistent to
calculate various metrics to compare the captions generated by Florence-2 with the
ground truth captions of the COCO 2014 Validation Dataset.

### Create Captions Using Florence-2
After starting Triton Inference Server, run the code to create the results file
[edit filepaths as needed]

```sh
cd model-repository/florence2
python validate.py \
    --coco-caption-dir "~/git/coco-caption" \
    --data-dir "~/data/ms_coco_2014/val2014"
```

On an RTX 4090 GPU, the 40,504 images in the validation dataset are captioned in 26mins
20 seconds. This equates to processing 25.6 images / second. The results file,
florence2_captions.json, is written to the `coco-caption/results` directory. This file
will be compared to the `coco-caption/annotation/captions_val2014.json` ground truth
captions in the next step.

### Calculate Metrics
The COCO-Caption repo makes it simple to calculate various metrics to compare the
Florence-2 generated caption against the validation data set.  Change directory to the
root of the repository and then run the following script.

```python
import argparse
from pathlib import Path
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def parse_cmd_line():
    parser = argparse.ArgumentParser(description="COCO Caption Evaluation")
    parser.add_argument("--coco-caption-dir", type=str, default="~/git/coco-caption")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cmd_line()

    coco_caption_dir = Path(args.coco_caption_dir).expanduser()
    annotation_file = coco_caption_dir / "annotations" / "captions_val2014.json"
    results_file = coco_caption_dir / "results" / "florence2_captions.json"

    coco = COCO(annotation_file)
    cocoRes = coco.loadRes(results_file)

    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()

    # print output evaluation scores
    for metric, score in cocoEval.eval.items():
        print(f"{metric}: {score:.3f}")

```

We compare the different metrics against Table 5 in the Florence-2 paper
(Generalist Models) and Table 1 in the MS COCO paper (Human Agreement for Image
Captioning).

| Metric   | Triton  |  Paper  | MS-COCO |
|----------|---------|---------|---------|
| BLEU1    |  0.830  |  N/A    | 0.880   |
| BLEU2    |  0.690  |  N/A    | 0.744   |
| BLEU3    |  0.555  |  N/A    | 0.603   |
| BLEU4    |  0.442  |  N/A    | 0.471   |
| METEOR   |  0.328  |  N/A    | 0.335   |
| ROUGE_L  |  0.630  |  N/A    | 0.626   |
| CIDEr    |  1.478  |  1.433  |  N/A    |

NOTE: No idea how you get CIDEr to be higher than 1, but it seems close