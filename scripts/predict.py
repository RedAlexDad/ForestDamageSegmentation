"""Inference script for predictions."""
import os
import argparse
import numpy as np

from src.data import load_image
from src.inference import Segmentor, predict_with_postprocess
from src.data.loader import SatelliteTileLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--postprocess", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from {args.model_path}...")
    predictor = Segmentor.from_path(args.model_path, args.threshold)

    os.makedirs(args.output_dir, exist_ok=True)

    loader = SatelliteTileLoader(args.input_dir)
    tile_paths = loader.load_tiles()
    print(f"Found {len(tile_paths)} tiles")

    for i, path in enumerate(tile_paths):
        print(f"Processing {i+1}/{len(tile_paths)}: {path}")

        img = loader.load_tile(path, normalize=True)
        x = loader.get_all_channels(img, n_channels=6)
        mask = loader.get_mask(img)

        if args.postprocess:
            pred = predict_with_postprocess(
                predictor.model,
                x,
                threshold=args.threshold
            )
        else:
            pred = predictor.predict(x)

        output_path = os.path.join(
            args.output_dir,
            os.path.basename(path).replace(".tif", "_pred.tif")
        )

        from skimage import io
        io.imsave(output_path, (pred * 65535).astype(np.uint16))

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()