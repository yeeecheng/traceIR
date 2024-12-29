import os
from argparse import ArgumentParser
from PIL import Image, ImageOps

def resize_image_to_resolution(input_image, resolution, reverse=True):
    width, height = input_image.size
    scale = resolution / min(width, height) if reverse else resolution / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64
    return ImageOps.fit(input_image, (new_width, new_height), method=Image.Resampling.LANCZOS)


def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=320, type=int)
    parser.add_argument("--indir", default='./resize/in', type=str)
    parser.add_argument("--outdir", default="./resize/out", type=str)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    for image_path in os.listdir(args.indir):
        input_image = Image.open(os.path.join(args.indir, image_path)).convert("RGB")
        input_image = resize_image_to_resolution(input_image, args.resolution, 'inpaint' not in image_path)
        input_image.save(os.path.join(args.outdir, image_path))


if __name__ == "__main__":
    main()