import pyiqa
import torch
import glob
import argparse
import os
from PIL import Image, ImageOps
from tqdm import tqdm


device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
musiq_metric = pyiqa.create_metric('musiq', device=device)
maniqa_metric = pyiqa.create_metric('maniqa', device=device)
clipiqa_metric = pyiqa.create_metric('clipiqa', device=device)


def calculate_score(result):

    musiqs = musiq_metric(result)
    maniqas = maniqa_metric(result)
    clipiqas = clipiqa_metric(result)

    return musiqs.item(), maniqas.item(), clipiqas.item()

def main(args):
    
    dir = args.result_dir
    folders = os.listdir(dir)
    pbar =  tqdm(range(len(folders)))

    musiqs, maniqas, clipiqas = [], [], []
    for i in pbar:
        musiq, maniqa, clipiqa = calculate_score(os.path.join(dir, folders[i]))
        musiqs.append(musiq)
        maniqas.append(maniqa)
        clipiqas.append(clipiqa)
    print(f'MUSIQ: {sum(musiqs)/len(musiqs):.4f}, ManIQA: {sum(maniqas)/len(maniqas):.4f}, CLIPIQA: {sum(clipiqas)/len(clipiqas):.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default='./validation_results')
    args = parser.parse_args()
    main(args)