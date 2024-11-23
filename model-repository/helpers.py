from io import BytesIO
import numpy as np
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoProcessor


class Florence2Processor:
    """Helper class to make debugging easier outside of Triton"""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            local_files=True,
            trust_remote_code=True,
        )
        self.pad_token_id = self.processor.tokenizer.pad_token_id

    def convert_np_to_str(self, prompt_np) -> str:
        try:
            prompt = prompt_np.reshape(-1)[0].decode("utf-8")
        except Exception as exc:
            raise ValueError(f"Failed to convert prompt input to str: {exc}")
        return prompt

    def convert_np_to_image(self, image_bytes_np) -> Image:
        try:
            image = Image.open(BytesIO(image_bytes_np.reshape(-1)[0])).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Failed to convert numpy bytes to PIL.Image: {exc}")
        return image

    def convert_bytes_to_np(self, bytes: bytes) -> np.ndarray:
        """Helper function for testing purposes to simulate Triton Inference Server's
        preferred method of shuttling numpy arrays around.

        Parameters
        ----------
        bytes : bytes
            Input bytes to be converted

        Returns
        -------
        np.ndarray
            Corresponding np.object_ array of shape [1, -1]

        Raises
        ------
        ValueError
        """
        try:
            array_np = np.array([bytes], dtype=np.object_).reshape(1, -1)
        except Exception as exc:
            raise ValueError(f"Failed to convert bytes to np: {exc}")
        return array_np

    def process(
        self,
        image_bytes_np: np.ndarray,
        task_prompt_np: np.ndarray,
        text_prompt_np: np.ndarray = None,
    ):
        try:
            image = self.convert_np_to_image(image_bytes_np)
            image_size = np.array([image.width, image.height]).reshape(1, -1)
            prompt = self.convert_np_to_str(task_prompt_np)
            if text_prompt_np:
                text_prompt = self.convert_np_to_str(text_prompt_np)
                prompt += text_prompt

        except Exception as exc:
            raise ValueError(exc)

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="np",
        )
        return (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
            image_size,
        )

    def parsed_answer(
        self,
        generated_ids: torch.Tensor,
        task_prompt_np: np.ndarray,
        image_size_np: np.ndarray,
    ):
        truncated_ids = generated_ids.reshape(-1)
        truncated_ids = truncated_ids[truncated_ids != self.pad_token_id]
        generated_text = self.processor.decode(truncated_ids)
        task_prompt = task_prompt_np.reshape(-1)[0].decode("utf-8")
        image_width, image_heigth = image_size_np.reshape(-1)

        parsed_text = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image_width, image_heigth)
        )

        return np.array([parsed_text], dtype=np.object_).reshape(1, -1)


class Florence2Model:
    def __init__(self, model_id: str, device: str, torch_dtype) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            # attn_implementation="flash_attention_2",
            torch_dtype=torch_dtype,
            use_safetensors=True,
            local_files_only=True,
            trust_remote_code=True,
        ).to(device)

        # Bit hacky, hopefully doesn't come back to bite
        # Copying value of processor.pad_token_id
        self.pad_token_id = 1

    def generate_ids(
        self,
        batch_input_ids: list[np.ndarray],
        batch_pixel_values: list[np.ndarray],
        batch_attention_mask: list[np.ndarray] = None,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        num_beams: int = 3,
        **generate_kwargs,
    ):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create padded batch of input_ids.
        input_ids = pad_sequence(
            [torch.from_numpy(in_ids).reshape(-1) for in_ids in batch_input_ids],
            batch_first=True,
            padding_value=self.pad_token_id,
        ).to(device=self.device)
        # Create batch of pixel values
        pixel_values = torch.from_numpy(np.vstack(batch_pixel_values)).to(
            device=self.device, dtype=self.torch_dtype
        )
        # Create batch of attention masks
        if batch_attention_mask:
            attention_mask = pad_sequence(
                [
                    torch.from_numpy(attn_msk).reshape(-1)
                    for attn_msk in batch_attention_mask
                ],
                batch_first=True,
                padding_value=0,
            ).to(device=self.device)
        else:
            attention_mask = None

        generated_ids = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            **generate_kwargs,
        )

        return generated_ids
