import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.utils import save_image
from PIL import Image, ImageFile
from pathlib import Path
from tqdm import tqdm
from function import DEVICE, test_transform, adjust_learning_rate, InfiniteSamplerWrapper, FlatFolderDataset, \
    train_transform
from network import Net



Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 测试函数
def eval(args):
    mes = "current pid: " + str(os.getpid())
    print(mes)
    logging.info(mes)
    model = Net(args)
    model.eval()
    model = model.to(DEVICE)
    content_dir = args.content
    style_dir = args.style
    content_dir = Path(content_dir)
    content_paths = [f for f in content_dir.glob('*')]
    style_dir = Path(style_dir)
    style_paths = [f for f in style_dir.glob('*')]

    tf = test_transform()
    if args.run_folder == True:
        print("0")
        i = 0
        for i, content_path in tqdm(enumerate(content_paths)):
            Ic = tf(Image.open(content_path).convert("RGB")).to(DEVICE)
            Ic = Ic.unsqueeze(dim=0)
            for j, style_path in tqdm(enumerate(style_paths)):
                Is = tf(Image.open(style_path).convert("RGB")).to(DEVICE)

                Is = Is.unsqueeze(dim=0)
                #                 print(Is.shape,Ic.shape)
                with torch.no_grad():
                    Ics = model(Ic, Is)

                #                 print(Ics)
                save_image(Ics[0], os.path.join(args.save_dir, f'{i}_{j}.jpg'))
                save_image(Ic[0], os.path.join(args.save_dirc, f'{i}_{j}.jpg'))
                save_image(Is[0], os.path.join(args.save_dirs, f'{i}_{j}.jpg'))


def main():
    main_parser = argparse.ArgumentParser(description="main parser")

    main_parser.add_argument("--pretrained", type=bool, default=True,
                             help="whether to use the pre-trained checkpoints")
    main_parser.add_argument("--requires_grad", type=bool, default=True,
                             help="set to True if the model requires model gradient")

    # 创建子解析器
    #     main_parser._subparsers
    #     subparsers = main_parser.add_subparsers(title="subcommands",dest="subcommmand")
    main_parser.add_argument("--train", help="training mode parser")

    main_parser.add_argument("--training", type=bool, default=False)
    main_parser.add_argument("--iterations", type=int, default=30000,
                             help="total training epochs (default: 160000)")
    main_parser.add_argument("--batch_size", type=int, default=4,
                             help="training batch size (default: 8)")
    main_parser.add_argument("--num_workers", type=int, default=8,
                             help="iterator threads (default: 8)")
    main_parser.add_argument("--lr", type=float, default=1e-4, help="the learning rate during training (default: 1e-4)")
    main_parser.add_argument("--content_folder", type=str,
                             default="/kaggle/input/coco-wikiart-nst-dataset-512-100000/content",
                             help="the root of content images, the path should point to a folder")
    main_parser.add_argument("--style_folder", type=str,
                             default="/kaggle/input/coco-wikiart-nst-dataset-512-100000/style",
                             help="the root of style images, the path should point to a folder")
    main_parser.add_argument("--log_interval", type=int, default=10000,
                             help="number of images after which the training loss is logged (default: 20000)")
    main_parser.add_argument("--save_dir", default='axperiments',
                             help='Directory to save the model')
    main_parser.add_argument("--save_dirc", default='content', help='Directory to save the model')
    main_parser.add_argument("--save_dirs", default='style', help='Directory to save the model')
    main_parser.add_argument("--log_dir", default='log',
                             help='Directory to save the log')
    main_parser.add_argument("--w_content1", type=float, default=12, help="the stage1 content loss weight")
    main_parser.add_argument("--w_content2", type=float, default=9, help="the stage2 content loss weight")
    main_parser.add_argument("--w_content3", type=float, default=7, help="the stage3 content loss weight")
    main_parser.add_argument("--w_remd1", type=float, default=2, help="the stage1 remd loss weight")
    main_parser.add_argument("--w_remd2", type=float, default=2, help="the stage2 remd loss weight")
    main_parser.add_argument("--w_remd3", type=float, default=2, help="the stage3 remd loss weight")
    main_parser.add_argument("--w_moment1", type=float, default=2, help="the stage1 moment loss weight")
    main_parser.add_argument("--w_moment2", type=float, default=2, help="the stage2 moment loss weight")
    main_parser.add_argument("--w_moment3", type=float, default=2, help="the stage3 moment loss weight")
    main_parser.add_argument("--color_on", type=str, default=True, help="turn on the color loss")
    main_parser.add_argument("--w_color1", type=float, default=0.25, help="the stage1 color loss weight")
    main_parser.add_argument("--w_color2", type=float, default=0.5, help="the stage2 color loss weight")
    main_parser.add_argument("--w_color3", type=float, default=1, help="the stage3 color loss weight")

    main_parser.add_argument("--run_folder", type=bool, default=True)
    main_parser.add_argument("--content", type=str, default="/kaggle/input/testimg/test/content",
                             help="content image you want to stylize")
    main_parser.add_argument("--style", type=str, default="/kaggle/input/testimg/test/style",
                             help="style image for stylization")

    args = main_parser.parse_args(args=[])
    #     print(args)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.save_dirc):
        os.mkdir(args.save_dirc)
    if not os.path.exists(args.save_dirs):
        os.mkdir(args.save_dirs)


    eval(args)  # 测试不参与损失的计算


if __name__ == "__main__":
    main()