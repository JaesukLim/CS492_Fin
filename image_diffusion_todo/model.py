from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class DiffusionModule(nn.Module):
    def __init__(self, network, var_scheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    def get_loss(self, x0, class_label=None, noise=None, pen_label=None):
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute noise matching loss.
        B = x0.shape[0]
        timestep = self.var_scheduler.uniform_sample_t(B, self.device)

        alpha_prod_t = self.var_scheduler.alphas_cumprod[timestep].view(B, 1, 1).expand(x0.shape)
        eps = torch.randn(x0.shape).to(x0.device)

        xt = alpha_prod_t.sqrt() * x0 + (1 - alpha_prod_t).sqrt() * eps
        eps_theta, pen_state = self.network(xt[:, :, 0:2], timestep, y=class_label)

        x0 = ((eps[:, :, 0:2] - eps_theta) ** 2).mean()
        # Calculating Pen State Loss
        # print(pen_state, pen_label)
        pen_loss = nn.BCELoss()(pen_state, pen_label)

        loss = x0 + 0.5 * pen_loss
        ######################
        return loss
    
    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 0.0,
    ):
        x_T = torch.randn([batch_size, 96, 3]).to(self.device)

        do_classifier_free_guidance = guidance_scale > 0.0

        if do_classifier_free_guidance:

            ######## TODO ########
            # Assignment 2-3. Implement the classifier-free guidance.
            # Specifically, given a tensor of shape (batch_size,) containing class labels,
            # create a tensor of shape (2*batch_size,) where the first half is filled with zeros (i.e., null condition).
            assert class_label is not None
            assert len(class_label) == batch_size, f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"
            null_conditions = torch.zeros_like(class_label).to(self.device)
            # class_label = torch.concat((null_conditions, class_label), dim=0)
            class_label = class_label.to(self.device)
            #######################

        traj = [x_T]
        for t in tqdm(self.var_scheduler.timesteps):
            x_t = traj[-1]
            if do_classifier_free_guidance:
                ######## TODO ########
                # Assignment 2. Implement the classifier-free guidance.
                noise_class, pen_state = self.network(x_t[:, :, 0:2], timestep=t.to(self.device), y=class_label)
                null_class, pen_state_null = self.network(x_t[:, :, 0:2], timestep=t.to(self.device), y=null_conditions)
                noise_pred = (1 + guidance_scale) * noise_class - guidance_scale * null_class
                #######################
            else:
                noise_pred, pen_state = self.network(
                    x_t[:, :, 0:2],
                    timesteps=t.to(self.device),
                    y=class_label,
                )

            x_t_prev = self.var_scheduler.step(x_t[:, :, 0:2], t, noise_pred, pen_state)

            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
            } 
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)
