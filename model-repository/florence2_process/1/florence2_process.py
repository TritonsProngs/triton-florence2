import base64
import importlib.util
import json
import numpy as np
import os

import triton_python_backend_utils as pb_utils

try:
    from ...helpers import Florence2Processor
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
    Florence2Processor = helpers_module.Florence2Processor


class TritonPythonModel:
    """
    Triton Inference Server deployment utilizing the python_backend for
    Florence-2-ft Processor
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
        model_id_or_dir = model_config["parameters"]["model_id_or_dir"]["string_value"]
        self.processor = Florence2Processor(model_id_or_dir)

    def error_tensors(self):
        return [
            # Output just the <s> and </s> tokens
            pb_utils.Tensor("INPUT_IDS", np.array([0, 2], dtype=np.int64)),
            # Output matching attention_mask
            pb_utils.Tensor("ATTENTION_MASK", np.ones((1, 2), dtype=np.int64)),
            # Output pixel values of all zeros
            pb_utils.Tensor(
                "PIXEL_VALUES", np.zeros((1, 3, 768, 768), dtype=np.float32)
            ),
            # Output just 1x1 for image size
            pb_utils.Tensor("IMAGE_SIZE", np.ones((1, 2), dtype=np.int64)),
        ]

    def execute(self, requests: list) -> list:
        batch_size = len(requests)
        self.logger.log_info(
            f"florence2_process.execute received {batch_size} requests"
        )
        responses = [None] * batch_size
        for batch_id, request in enumerate(requests):
            try:
                b64_image_np = pb_utils.get_input_tensor_by_name(
                    request, "INPUT_IMAGE"
                ).as_numpy()
                image_bytes = base64.b64decode(b64_image_np.reshape(-1)[0])
                image_bytes_np = np.array([image_bytes], np.object_).reshape(-1, 1)
                task_prompt_np = pb_utils.get_input_tensor_by_name(
                    request, "TASK_PROMPT"
                ).as_numpy()
                # Optional input
                text_prompt_np = pb_utils.get_input_tensor_by_name(
                    request, "TEXT_PROMPT"
                )
                self.logger.log_info(f"text_prompt_np is '{text_prompt_np}'")
                if text_prompt_np is not None:
                    text_prompt_np = text_prompt_np.as_numpy()

                input_ids, attention_mask, pixel_values, image_size = (
                    self.processor.process(
                        image_bytes_np, task_prompt_np, text_prompt_np
                    )
                )
            except Exception as exc:
                response = pb_utils.InferenceResponse(
                    output_tensors=self.error_tensors(),
                    error=pb_utils.TritonError(
                        f"Failed to get tokenize inputs from request: {exc}"
                    ),
                )
                responses[batch_id] = response
            else:
                response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("INPUT_IDS", input_ids),
                        pb_utils.Tensor("ATTENTION_MASK", attention_mask),
                        pb_utils.Tensor("PIXEL_VALUES", pixel_values),
                        pb_utils.Tensor("IMAGE_SIZE", image_size),
                    ]
                )
                responses[batch_id] = response

        return responses
