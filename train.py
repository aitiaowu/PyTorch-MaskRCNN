import bisect
import glob
import os
import re
import time
import random
import torch
from torch.utils.data import random_split
import pytorch_mask_rcnn as pmr
#from coco_Dataset import AgriRobotDataset
#from pytorch_mask_rcnn.datasets.coco_dataset import my_collate_fn
    
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda": 
        pmr.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
        
    # ---------------------- prepare data loader ------------------------------- #
    
    from torch.utils.data.sampler import SubsetRandomSampler
    data_dir = "/content/DATA"
    batch_size = 1
    dataset = pmr.datasets('coco', data_dir, train=True)

    # 定义训练集和验证集的大小，这里我们设定训练集占整个数据集的80%，验证集占10%
    total_len = len(dataset)
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    train_size = int(total_len * train_ratio)
    val_size = int(total_len * val_ratio)
    test_size = total_len - train_size - val_size  # Let test_size be the rest

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0,drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0,drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0, drop_last=True)

    #print(dir(train_dataset.dataset),len(train_dataset))
    
    args.warmup_iters = max(1000, len(train_loader))

    # -------------------------------------------------------------------------- #
    
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

        iter_train = pmr.train_one_epoch(model, optimizer, train_loader, val_loader, device, epoch, args)
        
        A = time.time() - A
        
        B = time.time()

        #eval_output, iter_eval = pmr.evaluate(model, val_dataset.dataset, device, args)
        #pmr.save_ckpt(model, optimizer, trained_epoch, args.ckpt_path, eval_info=str(eval_output))
        B = time.time() - B
    
        trained_epoch = epoch + 1
        print("training: {:.1f} s, evaluation: {:.1f} s".format(A, B))
        #pmr.collect_gpu_info("maskrcnn", [1 / iter_train, 1 / iter_eval])

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
    parser.add_argument("--ckpt-path", default="/content/drive/MyDrive/Study/Thesis/checkpoints/")
    parser.add_argument("--results")
    
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[6, 7])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--iters", type=int, default=-1, help="max iters per epoch, -1 denotes auto")
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")
    args = parser.parse_args()
    
    if args.lr is None:
        args.lr = 0.02 * 1 / 16 # lr should be 'batch_size / 16 * 0.02'
    if args.ckpt_path is None:
        args.ckpt_path = "./maskrcnn_{}.pth".format(args.dataset)
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_results.pth")
    
    main(args)
    
    