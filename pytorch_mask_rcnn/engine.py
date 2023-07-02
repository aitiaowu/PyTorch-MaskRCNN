import sys
import time

import torch
from .utils import save_ckpt
from .utils import Meter, TextArea
try:
    from .datasets import CocoEvaluator, prepare_for_coco
except:
    pass
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter()

def train_one_epoch(model, optimizer, data_loader, device, epoch, args):
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    model.train()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        num_iters = epoch * len(data_loader) + i
        image = image.squeeze(0).to(device)  # [C, H, W]
        target['masks'] = target['masks'].squeeze(0).to(device)  # [N, H, W]

        if num_iters <= args.warmup_iters:
            r = num_iters / args.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * args.lr_epoch
        
                   
        image = image.to(device)
        #print(image.shape)
        

        target = {k: v.to(device) for k, v in target.items()}
        
        S = time.time()
        
        losses = model(image, target)
        

        total_loss = sum(losses.values())
        print('iter',num_iters, 'total_loss:{:.5f}'.format(total_loss.item()))

        #tensorboard
        writer.add_scalar("Loss/all", total_loss.item(), num_iters)
        writer.flush()

        #saving ckpt
        if total_loss <= 0.003:
          save_ckpt(model, optimizer, num_iters + i, "/content/drive/MyDrive/Study/Thesis/checkpoints/model" + f'_{epoch}_{i}.pth')
          print('saving checkpoints')

        m_m.update(time.time() - S)
            
        S = time.time()
        total_loss.backward()
        b_m.update(time.time() - S)
        
        optimizer.step()
        optimizer.zero_grad()

        

        #eval_output, iter_eval = evaluate(model, data_loader, device, args)

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
        iter_eval = generate_results(model, data_loader, device, args)

    dataset = data_loader 
    iou_types = ["bbox", "segm"]
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
        
    return output, iter_eval
    
    
# generate results file   
@torch.no_grad()   
def generate_results(model, data_loader, device, args):
    iters = len(data_loader) if args.iters < 0 else args.iters
        
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
        #torch.cuda.synchronize()
        output = model(image)
        m_m.update(time.time() - S)
        
        prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
        coco_results.extend(prepare_for_coco(prediction))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
     
    A = time.time() - A 
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    torch.save(coco_results, args.results)
        
    return A / iters
    

