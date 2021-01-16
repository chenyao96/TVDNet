import torch
import torch.nn as nn
import torch.nn.functional as F


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        pred = self._tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class FocalLoss(nn.Module):
  def __init__(self):
    super(FocalLoss, self).__init__()

  def forward(self, pred, gt):
      pos_inds = gt.eq(1).float()
      neg_inds = gt.lt(1).float()
      neg_weights = torch.pow(1 - gt, 4)

      loss = 0

      pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
      neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

      num_pos  = pos_inds.float().sum()
      pos_loss = pos_loss.sum()
      neg_loss = neg_loss.sum()

      if num_pos == 0:
        loss = loss - neg_loss
      else:
        loss = loss - (pos_loss + neg_loss) / num_pos
      return loss


class VectorLoss(nn.Module):
    def __init__(self):
        super(VectorLoss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)

        #out[i][j][k] = input[index[i][j][k]] [j][k]  # if dim == 0
        #out[i][j][k] = input[i] [index[i][j][k]] [k]  # if dim == 1
        #out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

        feat = feat.gather(1, ind)
        # print(feat.shape)
        # print(feat)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target, target_vec):
        pred = self._tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # sub_vector = np.subtract(pred * mask, target * mask).to('CUDA')
        sub_vector = torch.sub(target * mask, pred * mask)
        loss = F.mse_loss(sub_vector, target_vec * mask, reduction='sum')
        loss_cos = (1-(torch.cosine_similarity(sub_vector, target_vec*mask, dim=0))).sum()
        loss = (loss/5000+loss_cos) / (mask.sum() + 1e-4)  # +loss_cos
        return loss


class LossAll(torch.nn.Module):
    def __init__(self):
        super(LossAll, self).__init__()
        self.L_hm = FocalLoss()
        self.L_off = RegL1Loss()
        self.L_wh = RegL1Loss()
        self.L_vector = VectorLoss()

    def forward(self, pr_decs, gt_batch):
        mid_vector_loss = self.L_vector(pr_decs['mid_point'], gt_batch['reg_mask'], gt_batch['ind'],
                                        gt_batch['next_cen'], gt_batch['next_vector'])
        hm_loss  = self.L_hm(pr_decs['hm'],  gt_batch['hm'])
        wh_loss  = self.L_wh(pr_decs['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        off_loss = self.L_off(pr_decs['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])
        loss_dec = hm_loss + off_loss + wh_loss + mid_vector_loss
        return loss_dec
