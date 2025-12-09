import argparse
import os
from typing import Dict

import cv2
import torch
from torch.utils.data import DataLoader

from arch import ResConvo
from data import transform as data_transform
from load_test import DataloadFromVideoTest


def parse_args():
    parser = argparse.ArgumentParser(description="360VSR inference")
    parser.add_argument("--input-video", required=True, help="Path to the input panoramic video (e.g., MP4)")
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Optional full path for the output video file. If omitted, the result is saved beside the input "
            "video with an '_sr' suffix."
        ),
    )
    parser.add_argument("--model-path", required=True, help="Path to a pretrained checkpoint")
    parser.add_argument("--scale", type=int, default=4, help="Super resolution scale factor")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help="Maximum number of frames to load (-1 to use every frame)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override the max frame window used during inference",
    )
    parser.add_argument("--fps", type=float, default=25.0, help="Output video frame rate")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run inference on",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    return parser.parse_args()


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if all(not k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_model(model_path: str, scale: int, device: torch.device) -> ResConvo:
    n_c = 128
    n_b = 10
    model = ResConvo(scale, n_c, n_b)
    state_dict = torch.load(model_path, map_location=device)
    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _resolve_output_path(output_path: str, input_path: str):
    if output_path:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return output_path

    video_dir, video_file = os.path.split(os.path.abspath(input_path))
    stem, _ = os.path.splitext(video_file)
    os.makedirs(video_dir, exist_ok=True)
    return os.path.join(video_dir, f"{stem}_sr.mp4")


def save_video(predictions: torch.Tensor, output_path: str, fps: float):
    frames = []
    for idx in range(predictions.shape[2]):
        frame = predictions[0, :, idx, :, :].clamp(0, 1)
        img = frame.mul(255).byte().permute(1, 2, 0).cpu().numpy()
        frames.append(img)

    if not frames:
        raise ValueError("No frames produced for output video")

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    for img in frames:
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    writer.release()


def run_inference(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    frame_window = args.window_size if args.window_size is not None else args.max_frames
    if frame_window is not None and frame_window < 0:
        frame_window = None
    dataset = DataloadFromVideoTest(
        args.input_video,
        args.scale,
        transform=data_transform(),
        max_frames=frame_window,
    )
    dataloader = DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=1, shuffle=False)

    model = load_model(args.model_path, args.scale, device)

    output_path = _resolve_output_path(args.output_path, args.input_video)

    with torch.no_grad():
        for lr, _ in dataloader:
            lr = lr.to(device)
            _, _, T, _, _ = lr.shape
            outputs = []
            init = True
            hid = None
            prediction = None
            for i in range(T - 2):
                f1 = lr[:, :, i, :, :]
                f2 = lr[:, :, i + 1, :, :]
                f3 = lr[:, :, i + 2, :, :]
                if init:
                    init_temp = torch.zeros_like(lr[:, 0:1, 0, :, :])
                    init_out = init_temp.repeat(1, args.scale * args.scale * 3, 1, 1)
                    hid = init_temp.repeat(1, 128, 1, 1)
                    hid, prediction = model(f1, f2, f3, hid, init_out, init)
                    init = False
                else:
                    hid, prediction = model(f1, f2, f3, hid, prediction, init)
                outputs.append(prediction)
            predictions = torch.stack(outputs, dim=2)
            save_video(predictions, output_path, args.fps)


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
