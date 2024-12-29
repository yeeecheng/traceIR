import pyiqa
import torch
import glob
import argparse
import os
from PIL import Image, ImageOps


device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
psnr_metric = pyiqa.create_metric('psnr', device=device)
ssim_metric = pyiqa.create_metric('ssim', device=device)
lpips_metric = pyiqa.create_metric('lpips', device=device)
# musiq_metric = pyiqa.create_metric('musiq', device=device)
# maniqa_metric = pyiqa.create_metric('maniqa', device=device)
# clipiqa_metric = pyiqa.create_metric('clipiqa', device=device)


def calculate_score(result, gt):
    psnrs = psnr_metric(result, gt)
    ssims = ssim_metric(result, gt)
    lpipss = lpips_metric(result, gt)
    # musiqs = musiq_metric(result, gt)
    # maniqas = maniqa_metric(result, gt)
    # clipiqas = clipiqa_metric(result, gt)

    # return psnrs.item(), ssims.item(), lpipss.item(), musiqs.item(), maniqas.item(), clipiqas.item()
    return psnrs.item(), ssims.item(), lpipss.item()

def main(args):
    gt_dir = args.gt_dir
    tar_dir = args.tar_dir

    gt_paths = glob.glob(f'{gt_dir}/*')

    psnrs, ssims, lpipss, musiqs, maniqas, clipiqas = [], [], [], [], [], []
    for gt_p in gt_paths:
        
        # input_image = Image.open(gt_p).convert("RGB")
        # input_image = resize_image_to_resolution(input_image)
        # input_image.save(gt_p)
        basename = os.path.basename(gt_p).split('.')[0]
        psnr, ssim, lpips = calculate_score(f'{tar_dir}/{basename}.jpg', gt_p)
        psnrs.append(psnr)
        ssims.append(ssim)
        lpipss.append(lpips)
    #     musiqs.append(musiq)
    #     maniqas.append(maniqa)
    #     clipiqas.append(clipiqa)
    # print(f'Score: PSNR: {sum(psnrs)/len(psnrs):.4f}, SSIM: {sum(ssims)/len(ssims):.4f}, LPIPS: {sum(lpipss)/len(lpipss):.4f}, MUSIQ: {sum(musiqs)/len(musiqs):.4f}, ManIQA: {sum(maniqas)/len(maniqas):.4f}, CLIPIQA: {sum(clipiqas)/len(clipiqas):.4f}')
    print(f'Score: PSNR: {sum(psnrs)/len(psnrs):.4f}, SSIM: {sum(ssims)/len(ssims):.4f}, LPIPS: {sum(lpipss)/len(lpipss):.4f}')

def resize_image_to_resolution(input_image, resolution=320, reverse=True):
    width, height = input_image.size
    scale = resolution / min(width, height) if reverse else resolution / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64
    return ImageOps.fit(input_image, (new_width, new_height), method=Image.Resampling.LANCZOS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, default='./experiment_dataset/gt_data')
    parser.add_argument("--tar_dir", type=str, default='./validation_results')
    args = parser.parse_args()

    main(args)