import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from PIL import Image
from .pooler import RoIAlign
from .utils import Matcher, BalancedPositiveNegativeSampler, roi_align
from .box_ops import BoxCoder, box_iou, process_box, nms


def fastrcnn_loss(class_logit, box_regression, label, regression_target):
    #print('class_logit_shape',class_logit.shape, 'label_shape',label.shape)
    #print("Unique labels:", torch.unique(label))
    #print("NaN in logits:", torch.isnan(class_logit).any())
    #print("Inf in logits:", torch.isinf(class_logit).any())
    #aaa
    classifier_loss = F.cross_entropy(class_logit, label)

    N, num_pos = class_logit.shape[0], regression_target.shape[0]
    box_regression = box_regression.reshape(N, -1, 4)
    box_regression, label = box_regression[:num_pos], label[:num_pos]
    box_idx = torch.arange(num_pos, device=label.device)

    box_reg_loss = F.smooth_l1_loss(box_regression[box_idx, label], regression_target, reduction='sum') / N

    return classifier_loss, box_reg_loss


def maskrcnn_loss(mask_logit, proposal, matched_idx, label, gt_mask):

    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat((matched_idx, proposal), dim=1)
    #print(roi)       
    M = mask_logit.shape[-1]
    #print(M)
    gt_mask = gt_mask[:, None].to(roi)
    
    image_height, image_width = 562, 1000  # 需要替换为实际值

    # 保留索引
    index = roi[:, 0]

    # 对ROI的坐标进行调整
    roi[:, 1] = torch.clamp(roi[:, 1], min=0, max=image_width)
    roi[:, 2] = torch.clamp(roi[:, 2], min=0, max=image_height)
    roi[:, 3] = torch.clamp(roi[:, 3], min=0, max=image_width)
    roi[:, 4] = torch.clamp(roi[:, 4], min=0, max=image_height)

    # 还原索引
    roi[:, 0] = index
    #print(gt_mask.shape, roi.shape)

    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]
    
    
    #     # 存储掩码
    # mask_path = f"/content/sample_data/mask_gt.png"  # 掩码文件保存路径，使用不同的文件名以区分不同的掩码
    # mask1 = gt_mask.squeeze(0).squeeze(0).cpu().numpy()  # 将张量转换为NumPy数组
    # mask1 = (mask1 * 255).astype(np.uint8)  # 将像素值从[0, 1]范围映射到[0, 255]范围，并转换为整数类型
    # mask1 = Image.fromarray(mask1)  # 创建PIL图像对象
    # mask1.save(mask_path)  # 保存掩码

    idx = torch.arange(label.shape[0], device=label.device)
    mask_loss = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target)
    #print(mask_loss.item())
    
    return mask_loss
    

class RoIHeads(nn.Module):
    def __init__(self, box_roi_pool, box_predictor,
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 score_thresh, nms_thresh, num_detections):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor
        
        self.mask_roi_pool = None
        self.mask_predictor = None
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)
        
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.num_detections = num_detections
        self.min_size = 1
        
    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_predictor is None:
            return False
        return True
        
    def select_training_samples(self, proposal, target):
        gt_box = target['boxes']
        gt_label = target['labels']
        # print('gt_label:',gt_label.shape, gt_label)
        # print('gt:',gt_box.shape)
        # print('proposal',proposal.shape)

        # proposal = torch.cat((proposal, gt_box))
        
        iou = box_iou(gt_box, proposal)
        pos_neg_label, matched_idx = self.proposal_matcher(iou)
        pos_idx, neg_idx = self.fg_bg_sampler(pos_neg_label)
        idx = torch.cat((pos_idx, neg_idx))
        
        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], proposal[pos_idx])
        proposal = proposal[idx]

        #print(proposal.shape)
        #print(matched_idx)
        matched_idx = matched_idx[idx]
        
        label = gt_label[matched_idx]
        num_pos = pos_idx.shape[0]
        label[num_pos:] = 0

        # print("pos_idx size:", pos_idx.shape)
        # print("neg_idx size:", neg_idx.shape)
        # print("Unique_labels_0:", torch.unique(label))
        # print("Unique_matched_idx:", torch.unique(matched_idx))


        return proposal, matched_idx, label, regression_target
    
    def fastrcnn_inference(self, class_logit, box_regression, proposal, image_shape):
        N, num_classes = class_logit.shape
        
        device = class_logit.device
        pred_score = F.softmax(class_logit, dim=-1)
        box_regression = box_regression.reshape(N, -1, 4)
        
        boxes = []
        labels = []
        scores = []
        for l in range(1, num_classes):
            score, box_delta = pred_score[:, l], box_regression[:, l]

            keep = score >= self.score_thresh
            box, score, box_delta = proposal[keep], score[keep], box_delta[keep]
            box = self.box_coder.decode(box_delta, box)
            
            box, score = process_box(box, score, image_shape, self.min_size)
            
            keep = nms(box, score, self.nms_thresh)[:self.num_detections]
            box, score = box[keep], score[keep]
            label = torch.full((len(keep),), l, dtype=keep.dtype, device=device)
            
            boxes.append(box)
            labels.append(label)
            scores.append(score)

        results = dict(boxes=torch.cat(boxes), labels=torch.cat(labels), scores=torch.cat(scores))
        return results
    
    def forward(self, feature, proposal, image_shape, target):
        if self.training:
            proposal, matched_idx, label, regression_target = self.select_training_samples(proposal, target)
        
        box_feature = self.box_roi_pool(feature, proposal, image_shape)
        #print(image_shape)
        class_logit, box_regression = self.box_predictor(box_feature)
        
        result, losses = {}, {}
        if self.training:
            classifier_loss, box_reg_loss = fastrcnn_loss(class_logit, box_regression, label, regression_target)
            losses = dict(roi_classifier_loss=classifier_loss, roi_box_loss=box_reg_loss)
        else:
            result = self.fastrcnn_inference(class_logit, box_regression, proposal, image_shape)
            
        if self.has_mask():
            if self.training:
                num_pos = regression_target.shape[0]
                
                mask_proposal = proposal[:num_pos]
                pos_matched_idx = matched_idx[:num_pos]
                mask_label = label[:num_pos]
                

                '''
                # -------------- critial ----------------
                box_regression = box_regression[:num_pos].reshape(num_pos, -1, 4)
                idx = torch.arange(num_pos, device=mask_label.device)
                mask_proposal = self.box_coder.decode(box_regression[idx, mask_label], mask_proposal)
                # ---------------------------------------
                '''
                
                if mask_proposal.shape[0] == 0:
                    losses.update(dict(roi_mask_loss=torch.tensor(0)))
                    return result, losses
            else:
                mask_proposal = result['boxes']
                
                if mask_proposal.shape[0] == 0:
                    result.update(dict(masks=torch.empty((0, 28, 28))))
                    return result, losses
   
            
            mask_feature = self.mask_roi_pool(feature, mask_proposal, image_shape)
            mask_logit = self.mask_predictor(mask_feature)
            
            if self.training:
                gt_mask = target['masks']

                #print(gt_mask.shape)
                mask_loss = maskrcnn_loss(mask_logit, mask_proposal, pos_matched_idx, mask_label, gt_mask)
                losses.update(dict(roi_mask_loss=mask_loss))
                #print(losses)
            else:
                label = result['labels']
                idx = torch.arange(label.shape[0], device=label.device)
                mask_logit = mask_logit[idx, label]

                mask_prob = mask_logit.sigmoid()
                result.update(dict(masks=mask_prob))
                
        return result, losses