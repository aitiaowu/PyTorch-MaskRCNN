
import argparse
import os
import time
import bisect
import glob
import os
import re
import time
import random
import torch

import torch
from torch.utils.data import random_split
import pytorch_mask_rcnn as pmr


def main(args):

    batch_size = 1
    dataset = pmr.datasets('coco', "/content/DATA", train=True)

    # 定义训练集和验证集的大小，这里我们设定训练集占整个数据集的80%，验证集占10%
    total_len = len(dataset)
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    train_size = int(total_len * train_ratio)
    val_size = int(total_len * val_ratio)
    test_size = total_len - train_size - val_size  # Let test_size be the rest

    # Perform the splits
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, total_len))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0,drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0,drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0, drop_last=True)

    
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    cuda = device.type == "cuda"

    if cuda: pmr.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))

    num_classes = 2
    model = pmr.maskrcnn_resnet50(False, num_classes).to(device)

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    #print(checkpoint["eval_info"])
    del checkpoint
    if cuda: torch.cuda.empty_cache()

    #print("\nevaluating...\n")

    B = time.time()
    eval_output, iter_eval = pmr.evaluate(model, test_dataset.dataset, device, args, generate=True)
    B = time.time() - B

    print(eval_output.get_AP())

    
    iters = 3

    for i, (image, target) in enumerate(train_loader):
        image = image.to(device)[0]
        #target = {k: v.to(device) for k, v in target.items()}

        with torch.no_grad():
            result = model(image)
        print(image.shape)

        pmr.show(image, result, dataset.classes, "/content/sample_data/{}.jpg".format(i))

        if i >= iters - 1:
            break
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--dataset", default="voc")
    parser.add_argument("--data-dir", default="/content/DATA")
    parser.add_argument("--ckpt-path", default="/content/drive/MyDrive/Study/Thesis/checkpoints/model_8_5499-72310.pth")
    parser.add_argument("--iters", type=int, default=1) # number of iterations, minus means the entire dataset
    args = parser.parse_args([]) # [] is needed if you're using Jupyter Notebook.

    args.use_cuda = True
    args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_results.pth")

    main(args)