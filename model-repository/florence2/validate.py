import argparse
import base64
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
import itertools
import json
from multiprocessing import cpu_count
from pathlib import Path
import requests


def parse_cmd_line():
    parser = argparse.ArgumentParser(description="Image Captioner")
    parser.add_argument("--coco-caption-dir", type=str, default="~/git/coco-caption")
    parser.add_argument("--data-dir", type=str, default="~/data/ms_coco_2014/val2014")

    return parser.parse_args()


class ImageCaptioner:
    def __init__(
        self,
        annotation_file: Path,
        data_dir: Path,
        model_url: str = "http://localhost:8000/v2/models/florence2/infer",
        max_workers: int = None,
    ):
        self.model_url = model_url
        self.max_workers = max_workers if max_workers else cpu_count()
        self.annotation_file = annotation_file

        self.images_path = {}
        annotations = json.load(open(annotation_file))
        for image_info in annotations["images"]:
            self.images_path[image_info["id"]] = str(data_dir / image_info["file_name"])

    @staticmethod
    def read_image(image_path: str) -> str:
        """Read an image and return it as a base64 encoded string

        Parameters
        ----------
        image_path : str
            Path to the image file

        Returns
        -------
        str
            base64 encoded image
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def get_images_b64(
        self, image_ids: list[int]
    ) -> tuple[dict[int, str], dict[int, str]]:
        """Get the base64 encoded images for a list of image ides using a process pool

        Parameters
        ----------
        image_ids : list[int]
            List of image ids from MS COCO Validation 2014 dataset

        Returns
        -------
        tuple[dict[int, str], dict[int, str]]
            Tuple of dictionaries with the base64 encoded images and errors
        """
        images_b64 = {}
        errors = {}
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = {}
            for image_id in image_ids:
                futures[
                    executor.submit(
                        self.read_image,
                        self.images_path[image_id],
                    )
                ] = image_id
            for future in as_completed(futures):
                image_id = futures[future]
                try:
                    images_b64[image_id] = future.result()
                except Exception as e:
                    errors[image_id] = str(e)
        return images_b64, errors

    def get_captions(
        self, results_file: Path, task_prompt: str = "<CAPTION>", chunk_size: int = 512
    ) -> list[str]:
        """Get the captions for the images in the MS COCO Validation 2014 dataset

        Parameters
        ----------
        results_file : Path
            Path to the file where the results will be saved
        task_prompt : str, optional
            The task prompt to use for the model. Default is "<CAPTION>"
        chunk_size : int, optional
            Number of images to process at a time. Default is 512

        Returns
        -------
        list[str]
            List of errors that occurred while processing the images
        """
        start_time = datetime.now()
        results = []
        errors = {}
        # Process in chunks
        for i, image_ids in enumerate(itertools.batched(self.images_path, chunk_size)):
            print(
                f"{datetime.now()} Starting on chunk {i+1} that has {len(image_ids)} images"
            )
            images_b64, batch_errors = self.get_images_b64(image_ids)
            errors.update(batch_errors)
            print(
                f"{datetime.now()} Finished getting {len(images_b64):,} images with {len(batch_errors):,} errors"
            )
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for image_id, image_b64 in images_b64.items():
                    inference_json = {
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
                            },
                        ]
                    }
                    future = executor.submit(
                        requests.post,
                        url=self.model_url,
                        json=inference_json,
                    )
                    futures[future] = image_id
                for future in as_completed(futures):
                    image_id = futures[future]
                    try:
                        response_json = future.result().json()
                    except Exception as e:
                        print(f"{image_id=:}, {e}")
                    else:
                        if "error" not in response_json:
                            output = json.loads(response_json["outputs"][0]["data"][0])
                            caption = output[task_prompt]
                            results.append({"image_id": image_id, "caption": caption})
                        else:
                            print(f"{image_id=:}, {response_json['error']}")
        end_time = datetime.now()
        print(f"Total time: {end_time - start_time}")
        delta = (end_time - start_time).total_seconds()
        n_images = len(results)
        print(f"Processed {n_images:,} images at {n_images/delta:.2f} images/sec")

        # Print out the results
        with open(results_file, "w") as f:
            json.dump(results, f)
        return errors


def main():
    args = parse_cmd_line()

    coco_caption_dir = Path(args.coco_caption_dir).expanduser()
    data_dir = Path(args.data_dir).expanduser()
    annotation_file = coco_caption_dir / "annotations" / "captions_val2014.json"
    results_file = coco_caption_dir / "results" / "florence2_captions.json"

    ic = ImageCaptioner(annotation_file=annotation_file, data_dir=data_dir)
    errors = ic.get_captions(results_file, chunk_size=811)
    # Print out some errors if we have any
    for i, (image_id, error) in enumerate(errors.items()):
        print(f"{image_id=:}, {error=}")
        if i > 20:
            break


if __name__ == "__main__":
    main()
