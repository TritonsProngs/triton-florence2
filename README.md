# triton-florence2
A Triton Inference Server model repository hosting the
[Florence-2 model](https://huggingface.co/mrhendrey/Florence-2-large-ft-safetensors).
This is the fine-tuned version of the model that supports the following tasks:

* Caption
* Detailed Caption
* More Detailed Caption
* Caption to Phrase Grounding (requires a text prompt)
* Object Detection
* Dense Region Caption
* Region Proposal
* OCR
* OCR with Region

## Running Tasks
Running tasks is orchestrated using [Taskfile.dev](https://taskfile.dev)

### Taskfile Instructions
Tasks are defined in `Taskfile.yml`. Create a task.env at the root of the project
to define environment overrides.

### Tasks Overview
The `Taskfile.yml` includes the following tasks:

* `triton-start`
* `triton-stop`
* `model-import`
* `build-execution-env-all`
* `build-*-env` (with options: `florence2_model` & `florence2_process`)

### Task Descriptions

#### `triton-start`

Starts the Triton server.

```sh
task triton-start
```

#### `triton-stop`

Stops the Triton server.

```sh
task triton-stop
```

#### `model-import`

Import model files from huggingface

```sh
task model-import
```

#### `task build-execution-env-all`

Builds all the conda pack environments used by Triton

```sh
task build-execution-env-all
```

#### `task build-*-env`

Builds specific conda pack environments used by Triton

```sh
#Example 
task build-florence2_model-env
```

#### `Complete Order`

Example of running multiple tasks to stage items needed to run Triton Server

```sh
task build-execution-env-all
task model-import
task triton-start
# Tail logs of running containr
docker logs -f $(docker ps -q --filter "name=triton-inference-server")
```

## NVIDIA's Model Analyzer
The Triton Inference Server SDK container includes the `model-analyzer` which can
identify optimal configurations (batch size, batch delays, number of instances, etc).
This is driven by the [`model-analyzer.yaml'](model-analyzer.yaml) file.

Be sure that you have run `task triton-start` as we leverage using the `remote` option
for model serving in the yaml file.

### Launch Triton Inference Server Docker Container

```sh
docker run --rm -it --net host --shm-size 1gb \
-v ./:/workspace/triton-florence2 \
nvcr.io/nvidia/tritonserver:24.10-py3-sdk
```

### Run model-analyzer
Inside of the sdk container, run the following command to profile the GPU
(florence2_model) and the CPU (florence2_process) components first.

```sh
model-analyzer profile \
  --model-repository triton-florence2/model-repository \
  --triton-launch-mode=remote \
  --output-model-repository-path triton-florence2/model-analyzer-output/brute_both \
  -f triton-florence2/model-analyzer-both.yaml
```

After this has run, do the following command to profile the ensemble deployment
(florence2) that is composed of the other two.

```sh
model-analyzer profile \
  --model-repository triton-florence2/model-repository \
  --triton-launch-mode=remote \
  --output-model-repository-path triton-florence2/model-analyzer-output/ensemble \
  -f triton-florence2/model-analyzer-ensemble.yaml \
  --override_output_model_repository
```