import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from detr.main import build_ACT_model, build_optimizer
from policies.common.wrapper import TemporalEnsembling


class HITPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, self._args = build_ACT_model(args_override)
        self.model = model
        self.feature_loss_weight = args_override["feature_loss_weight"] if "feature_loss_weight" in args_override else 0.0
        try:
            self.state_idx = args_override["state_idx"]
            self.action_idx = args_override["action_idx"]
        except:
            self.state_idx = None
            self.action_idx = None
        try:
            if args_override["temporal_agg"]:
                self.temporal_ensembler = TemporalEnsembling(
                    args_override["chunk_size"],
                    args_override["action_dim"],
                    args_override["max_timesteps"],
                )
        except Exception as e:
            print(e)
            print("The above Exception can be ignored when training instead of evaluating.")

    # TODO: 使用装饰器在外部进行包装
    def reset(self):
        if self.temporal_ensembler is not None:
            self.temporal_ensembler.reset()
    
    def __call__(self, qpos, image, actions=None, is_pad=None, task_emb=None):
        if self.state_idx is not None:
            qpos = qpos[:, self.state_idx]
        
        # TODO: imagenet norm, move this outside
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            if self.action_idx is not None:
                actions = actions[:, :, self.action_idx]
            actions = actions[:, :self.model.num_queries]
            assert is_pad is not None, "is_pad should not be None"
            is_pad = is_pad[:, : self.model.num_queries]

            a_hat, _, hs_img_dict = self.model(qpos, image, task_emb)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            if self.model.feature_loss:
                loss_dict["feature_loss"] = F.mse_loss(hs_img_dict["hs_img"], hs_img_dict["src_future"]).mean()
                loss_dict["loss"] = loss_dict["l1"] + self.feature_loss_weight * loss_dict["feature_loss"]
            else:
                loss_dict["loss"] = loss_dict["l1"]
            return loss_dict
        else:  # inference time
            a_hat, _, _ = self.model(qpos, image, task_emb) # no action, sample from prior
            if self.temporal_ensembler is None:
                return a_hat
            else:
                a_hat_one = self.temporal_ensembler.update(a_hat)
                a_hat[0][0] = a_hat_one
            return a_hat

    def configure_optimizers(self):
        self.optimizer = build_optimizer(self.model, self._args)
        return self.optimizer
