import sys
import time
import os
import numpy as np
from PIL import Image
import torch
from .utils import save_ckpt
from .utils import Meter, TextArea
try:
    from .datasets import CocoEvaluator, prepare_for_coco
except:
    pass
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter()

def plot_image_and_annotations(image, target):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # 画出 bounding boxes
    for box in target['boxes']:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
    # 如果你的 target 中包含 masks，你也可以画出 masks：
    for mask in target['masks']:
        ax.imshow(mask, alpha=0.5)
    imgid = target['image_id']

        # 如果你想保存图片，你可以使用 plt.savefig：
    plt.savefig("/content/sample_data"+ str(imgid.item()) + ".png")

def train_one_epoch(model, optimizer, data_loader, val_loader, device, epoch, args):
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters
    accum_iter = 4
    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    model.train()
    A = time.time()
    min_val_loss = float('inf')
    best_model_path = None  # 保存最佳模型的路径
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        num_iters = epoch * iters + i
        if num_iters <= args.warmup_iters:
            r = num_iters / args.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = 5 * r * args.lr_epoch

        # target['masks'] = target['masks'].squeeze(0)
        # target['boxes'] = target['boxes'].squeeze(0)
        # image = image.squeeze(0).permute(1, 2, 0).numpy()
        # plot_image_and_annotations(image, target)
        # a
        image = image.squeeze(0).to(device)  # [C, H, W]
        target['masks'] = target['masks'].squeeze(0).to(device)  # [N, H, W]  

        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        S = time.time()
        #print(image.shape,target['masks'].shape,target['boxes'].shape)

        losses = model(image, target)

        # ca_weights = model.backbone.body.layer3.cbam.ca_weights.detach().cpu().numpy()
        # sa_weights = model.backbone.body.layer3.cbam.sa_weights.detach().cpu().numpy()
        # np.save(f'/content/sample_data/ca_weights_{i}.npy', ca_weights)
        # np.save(f'/content/sample_data/sa_weights_{i}.npy', sa_weights)

        total_loss = sum(losses.values())
        #total_loss = sum(losses.values())/accum_iter
        print('iter',num_iters, 'total_loss:{:.5f}'.format(total_loss.item()))


        m_m.update(time.time() - S)
            
        S = time.time()

        total_loss.backward()
        b_m.update(time.time() - S)

        # weights update gradient accumulation
        if ((i + 1) % accum_iter == 0) or (i + 1 == len(data_loader)):
            optimizer.step()
            optimizer.zero_grad()

        # validation
        if ((i + 1) % 500 == 0):
          _,val_loss = generate_results(model, val_loader, device, args)
          print('val_loss',val_loss.item())
          writer.add_scalar("val_loss", val_loss, num_iters)
          if val_loss < min_val_loss:
                min_val_loss = val_loss
                model_path = "/content/drive/MyDrive/Study/Thesis/checkpoints/model_cbam" + str(epoch) + '.pth'
                
                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)

                #save_ckpt(model, optimizer, num_iters + i, model_path)
                best_model_path = model_path  # 更新最佳模型的路径
                print('Best model saved at', best_model_path)

        #tensorboard
        writer.add_scalar("train_loss", total_loss.item(), num_iters)
        
        writer.flush()

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
           
    A = time.time() - A
    print('Finish the epoch with iter:', i)
    #print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg,1000*b_m.avg))
    return A / iters, total_loss.item()
            

def evaluate(model, data_loader, device, args, generate=True):
    iter_eval = None
    if generate:
        iter_eval = Genrate(model, data_loader, device, args)

    dataset = data_loader 
    iou_types = ["bbox", "segm"]
    #iou_types = ["segm"]
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types)

    results = torch.load(args.results, map_location="cpu")


    S = time.time()
    coco_evaluator.accumulate(results)
    print("accumulate: {:.1f}s".format(time.time() - S))

    # collect outputs of buildin function print
    temp = sys.stdout
    sys.stdout = TextArea()

    coco_evaluator.summarize()

    output = sys.stdout
    sys.stdout = temp 
    #print(output)   
    return output, iter_eval
    
    
# generate results file   
@torch.no_grad()   
def generate_results(model, data_loader, device, args):
    iters = 600 if args.iters < 0 else args.iters
    val_loss = 0    
    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()

        image = image.squeeze(0).to(device)  # [C, H, W]
        target['masks'] = target['masks'].squeeze(0).to(device)  # [N, H, W]

        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        S = time.time()
        output = model(image)
        

        m_m.update(time.time() - S)
        
        prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}

        coco_results.extend(prepare_for_coco(prediction))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
    model.train()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()

        image = image.squeeze(0).to(device)  # [C, H, W]
        target['masks'] = target['masks'].squeeze(0).to(device)  # [N, H, W]      

        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        S = time.time()
        losses = model(image, target)
        total_loss = sum(losses.values())
        val_loss += total_loss

        m_m.update(time.time() - S)
        
        prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
        coco_results.extend(prepare_for_coco(prediction))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break


    A = time.time() - A 
    #print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    torch.save(coco_results, args.results)
        
    return A / iters , val_loss/iters
    
@torch.no_grad()   
def Genrate(model, data_loader, device, args):
    iters = 600 if args.iters < 0 else args.iters
    val_loss = 0    
    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()

        image = image.squeeze(0).to(device)  # [C, H, W]
        target['masks'] = target['masks'].squeeze(0).to(device)  # [N, H, W]
        #imgid = target['image_id']
        #             # 存储掩码
        # mask_path = "/content/sample_data/mask_gt_"+ imgid + ".png"  # 掩码文件保存路径，使用不同的文件名以区分不同的掩码
        # mask1 = target['masks'].squeeze(0).cpu().numpy()
        # mask1 = (mask1 * 255).astype(np.uint8)  # 将像素值从[0, 1]范围映射到[0, 255]范围，并转换为整数类型
        # #print(mask1.shape)  # 将张量转换为NumPy数组
        # #a
        # mask1 = Image.fromarray(mask1)  # 创建PIL图像对象
        # mask1.save(mask_path)  # 保存掩码
        
        ####
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        S = time.time()
        output = model(image)


        
        ####test masks prediction####


        m_m.update(time.time() - S)
        
        prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}

        # print("\n prediction shape")
        # print(prediction[target["image_id"].item()]['boxes'].shape,prediction[target["image_id"].item()]['masks'].shape)
        
        coco_results.extend(prepare_for_coco(prediction))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
    A = time.time() - A 
    #print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    torch.save(coco_results, args.results)
        
    return A / iters





