import importlib.util
import json
import numpy as np
import os
import torch

import triton_python_backend_utils as pb_utils

try:
    from ...helpers import Florence2Model, Florence2Processor
except:
    # Terrible Hack to get this to work in Triton Inference Server container
    # Get the absolute path for this file, then back up appropriate number of
    # directories to get to where helpers.py is located
    helper_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    helper_path = f"{helper_dir}/helpers.py"
    spec = importlib.util.spec_from_file_location("helpers", helper_path)
    helpers_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers_module)
    Florence2Model = helpers_module.Florence2Model
    Florence2Processor = helpers_module.Florence2Processor


class TritonPythonModel:
    """
    Triton Inference Server deployment utilizing the python_backend for
    Florence-2-ft model
    """

    def initialize(self, args):
        """
        Initialize the Triton Inference deployment

        Parameters
        ----------
        args : dict
            Command-line arguments for launching Triton Inference Server
        """
        self.logger = pb_utils.Logger
        self.model_config = model_config = json.loads(args["model_config"])

        # Get model location
        model_cache_dir = os.environ.get(
            "MODEL_CACHE_DIR", os.path.join(args["model_repository"], "models")
        )
        model_subdirectory = self.model_config["parameters"]["model_subfolder"][
            "string_value"
        ]
        self.model_path = os.path.abspath(
            os.path.join(model_cache_dir, model_subdirectory)
        )

        # Get GPU/CPU related things
        if args["model_instance_kind"] == "GPU" and torch.cuda.is_available():
            device_id = args["model_instance_device_id"]
            device_id = device_id if device_id else 0
            self.device = f"cuda:{device_id}"
            self.torch_dtype = torch.float16
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32

        # Load the processor & model
        self.processor = Florence2Processor(self.model_path)
        self.model = Florence2Model(self.model_path, self.device, self.torch_dtype)

    def error_tensors(self):
        error_bytes = json.dumps({"ERROR": "error"}).encode("utf-8")
        return [
            # Output simple error JSON
            pb_utils.Tensor("PARSED_JSON", np.array([error_bytes], dtype=np.object_)),
        ]

    def input_tensor_to_np(self, request, input_name):
        return pb_utils.get_input_tensor_by_name(request, input_name).as_numpy()

    def process_request(self, request):
        task_prompt = self.input_tensor_to_np(request, "TASK_PROMPT")
        input_ids = self.input_tensor_to_np(request, "INPUT_IDS")
        attention_mask = self.input_tensor_to_np(request, "ATTENTION_MASK")
        pixel_values = self.input_tensor_to_np(request, "PIXEL_VALUES")
        image_size = self.input_tensor_to_np(request, "IMAGE_SIZE")

        return task_prompt, input_ids, attention_mask, pixel_values, image_size

    def execute(self, requests: list) -> list:
        batch_size = len(requests)
        self.logger.log_info(f"florence2_model.execute received {batch_size} requests")
        responses = [None] * batch_size
        valid_requests = []
        batch_task_prompts = []
        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = []
        batch_image_size = []
        for batch_id, request in enumerate(requests):
            try:
                task_prompt, input_ids, attention_mask, pixel_values, image_size = (
                    self.process_request(request)
                )
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    output_tensors=self.error_tensors(),
                    error=pb_utils.TritonError(
                        f"Failed to get inputs from request: {exc}"
                    ),
                )
                responses[batch_id] = response
            else:
                valid_requests.append(batch_id)
                batch_task_prompts.append(task_prompt)
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_pixel_values.append(pixel_values)
                batch_image_size.append(image_size)

        try:
            generated_ids = self.model.generate_ids(
                batch_input_ids=batch_input_ids,
                batch_pixel_values=batch_pixel_values,
                batch_attention_mask=batch_attention_mask,
            )
        except Exception as exc:
            # ENTIRE BATCH FAILED. Need send errors to all
            for batch_id in valid_requests:
                response = pb_utils.InferenceResponse(
                    output_tensors=self.error_tensors(),
                    error=pb_utils.TritonError(
                        f"Entire batch failed on model.generate_ids: {exc}",
                    ),
                )
                responses[batch_id] = response
            return responses

        # Loop through batch results to create individual responses
        for batch_id, gen_ids, task_prompt, image_size in zip(
            valid_requests, generated_ids, batch_task_prompts, batch_image_size
        ):
            parsed_answer = self.processor.parsed_answer(
                gen_ids, task_prompt, image_size
            ).reshape(-1)[0]
            parsed_answer = json.dumps(parsed_answer)
            response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "PARSED_JSON",
                        np.array([parsed_answer], dtype=np.object_).reshape(1, -1),
                    )
                ]
            )
            responses[batch_id] = response

        return responses
