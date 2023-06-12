import bisect
import glob
import os
import re
import time
import random
import torch

import pytorch_mask_rcnn as pmr
#from coco_Dataset import AgriRobotDataset
    
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda": 
        pmr.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
        
    # ---------------------- prepare data loader ------------------------------- #
    
    from torch.utils.data.sampler import SubsetRandomSampler

    batch_size = 1
    dataset = pmr.datasets('coco', "/content/DATA", train=True)

    dataset_size = len(dataset)  # 总的数据集大小
    indices = list(range(dataset_size))  # 生成包含所有索引的列表
    random.shuffle(indices)  # 随机打乱索引列表

    # 设置训练集和验证集的比例
    train_ratio = 0.8  # 训练集占80%
    val_ratio = 0.1  # 验证集占10%，剩下的10%是测试集

    # 计算各个集合的大小
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)

    # 切分索引列表
    train_indices = indices[:train_size]  # 前80%是训练集
    val_indices = indices[train_size:(train_size + val_size)]  # 接下来10%是验证集
    test_indices = indices[(train_size + val_size):]  # 剩下的是测试集
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create train/val/test dataloader
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0,drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0,drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=0, drop_last=True)

    args.warmup_iters = max(1000, len(train_loader))
    
    # -------------------------------------------------------------------------- #
    print(len(train_loader))
    #print(args)
    num_classes = 1 + 1 # including background class
    model = pmr.maskrcnn_resnet50(True, num_classes).to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_lambda = lambda x: 0.1 ** bisect.bisect(args.lr_steps, x)
    
    start_epoch = 0
    
    # find all checkpoints, and load the latest checkpoint
    prefix, ext = os.path.splitext(args.ckpt_path)
    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device) # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epochs"]
        del checkpoint
        torch.cuda.empty_cache()

    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))
    
    # ------------------------------- train ------------------------------------ #
        
    for epoch in range(start_epoch, args.epochs):
        print("\nepoch: {}".format(epoch + 1))
            
        A = time.time()
        args.lr_epoch = lr_lambda(epoch) * args.lr
        print("lr_epoch: {:.5f}, factor: {:.5f}".format(args.lr_epoch, lr_lambda(epoch)))
        iter_train = pmr.train_one_epoch(model, optimizer, train_loader, device, epoch, args)
        A = time.time() - A
        
        B = time.time()
        eval_output, iter_eval = pmr.evaluate(model, val_loader, device, args)
        B = time.time() - B

        trained_epoch = epoch + 1
        print("training: {:.1f} s, evaluation: {:.1f} s".format(A, B))
        pmr.collect_gpu_info("maskrcnn", [1 / iter_train, 1 / iter_eval])
        print(eval_output.get_AP())

        pmr.save_ckpt(model, optimizer, trained_epoch, args.ckpt_path, eval_info=str(eval_output))

        # it will create many checkpoint files during training, so delete some.
        prefix, ext = os.path.splitext(args.ckpt_path)
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        n = 10
        if len(ckpts) > n:
            for i in range(len(ckpts) - n):
                os.system("rm {}".format(ckpts[i]))
        
    # -------------------------------------------------------------------------- #

    print("\ntotal time of this training: {:.1f} s".format(time.time() - since))
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true")
    
    parser.add_argument("--dataset", default="coco", help="coco or voc")
    parser.add_argument("--data-dir", default="/content/DATA")
    parser.add_argument("--ckpt-path")
    parser.add_argument("--results")
    
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[6, 7])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10, help="max iters per epoch, -1 denotes auto")
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")
    args = parser.parse_args()
    
    if args.lr is None:
        args.lr = 0.02 * 1 / 16 # lr should be 'batch_size / 16 * 0.02'
    if args.ckpt_path is None:
        args.ckpt_path = "./maskrcnn_{}.pth".format(args.dataset)
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_results.pth")
    
    main(args)
    
    