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


# sys.argv = ["notebook_name.py", "subcommand1"]

def train(args):
    logging.basicConfig(filename='training.log',
                        format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    mes = "current pid: " + str(os.getpid())  # 返回当前进度的ID并且将其转换为文本型打印
    print(mes)
    logging.info(mes)  # 记录日志信息
    model = Net(args)
    model.train()
    device_ids = [0]  # 多张显卡的使用0为2号卡，1为3号卡
    model = nn.DataParallel(model, device_ids)
    model = model.to(DEVICE)

    tf = train_transform()
    content_dataset = FlatFolderDataset(args.content_folder, tf)
    style_dataset = FlatFolderDataset(args.style_folder, tf)
    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.num_workers))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.num_workers))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    i = 0
    for img_index in range(args.iterations):
        print("iteration :", img_index + 1)
        optimizer.zero_grad()
        Ic = next(content_iter).to(DEVICE)
        Is = next(style_iter).to(DEVICE)
        loss, Ics = model(Ic, Is)
        print(f'现在的损失为：{loss}')
        loss.sum().backward()

        # plot_grad_flow(GMMN.named_parameters())
        optimizer.step()  # 更新模型参数
        if i % 1000 == 0:
            save_image(Ics[0], os.path.join(args.save_dir, f'{i}.jpg'))
            save_image(Ic[0], os.path.join(args.save_dirc, f'{i}.jpg'))
            save_image(Is[0], os.path.join(args.save_dirs, f'{i}.jpg'))

        i = i + 1
        if (img_index + 1) % args.log_interval == 0:
            print("saving...")
            mes = "iteration: " + str(img_index + 1) + " loss: " + str(loss.sum().item())
            logging.info(mes)
            model.module.save_ckpts()  # 将训练的模型数据保存
            adjust_learning_rate(optimizer, img_index, args)  # 调整学习率


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

        train(args)



if __name__ == "__main__":
    main()