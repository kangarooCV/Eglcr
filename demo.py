import sys

sys.path.append('core')
DEVICE = 'cuda'
# DEVICE = 'cpu'
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from eglcr_stereo import EGLCRStereo
from utils.utils import InputPadder
from PIL import Image
import os
import matplotlib.pyplot as plt


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[:, :, :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def save_pfm(filename, image, scale=1):

    file = open(filename, "w")
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:# color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale
        file.write('%f\n' % scale)
        image.tofile(file)

def demo(args):
    model = torch.nn.DataParallel(EGLCRStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt, map_location="cpu"), strict=False)
    model.to(DEVICE)
    model = model.module
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))

        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")
        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images)), ncols=70):

            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32, mode="md")
            image1, image2 = padder.pad(image1, image2)
            disp, edge = model(image1.to(DEVICE), image2.to(DEVICE), iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp)
            edge = padder.unpad(edge)

            disp[disp < 0.] = 0
            disp = disp.cpu().squeeze().numpy()
            edge = torch.sigmoid(edge).cpu().squeeze().numpy()

            file_stem = imfile1.split(os.sep)[-2]
            plt.imsave(output_directory / f"{file_stem}_edge.png", edge)
            plt.imsave(output_directory / f"{file_stem}.png", disp, cmap='jet')

            # save_disp = np.flipud(disp)
            # save_pfm(output_directory / f"disp0{file_stem}.pfm", save_disp, scale=1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=r'checkpoint/pretrain.pth')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')

    # middle bury
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
                        default=r"imgs/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
                        default=r"imgs/*/im1.png")

    parser.add_argument('--output_directory', help="directory to save output", default="output")
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=22, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")

    args = parser.parse_args()
    Path(args.output_directory).mkdir(exist_ok=True, parents=True)
    demo(args)