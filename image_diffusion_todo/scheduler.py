from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class BaseScheduler(nn.Module):
    def __init__(
        self, num_train_timesteps: int, beta_1: float, beta_T: float, mode="linear"
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts

class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode="linear",
        sigma_type="small",
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
    
        # sigmas correspond to $\sigma_t$ in the DDPM paper.
        self.sigma_type = sigma_type
        if sigma_type == "small":
            # when $\sigma_t^2 = \tilde{\beta}_t$.
            alphas_cumprod_t_prev = torch.cat(
                [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
            )
            sigmas = (
                (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            ) ** 0.5
        elif sigma_type == "large":
            # when $\sigma_t^2 = \beta_t$.
            sigmas = self.betas ** 0.5

        self.register_buffer("sigmas", sigmas)

    def step(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor, pen_state: torch.Tensor):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.

        Input:
            x_t (`torch.Tensor [B,C,H,W]`): samples at arbitrary timestep t.
            t (`int`): current timestep in a reverse process.
            eps_theta (`torch.Tensor [B,C,H,W]`): predicted noise from a learned model.
        Ouptut:
            sample_prev (`torch.Tensor [B,C,H,W]`): one step denoised sample. (= x_{t-1})
        """

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM reverse step.
        # eps_factor = (1 - self.alphas[t]) / (1 - self.alphas_cumprod[t]).sqrt()
        #
        # z = torch.randn(x_t.shape).to(x_t.device)
        # # x_t_prev = torch.normal((xt - eps_factor * eps_theta) / extract(self.var_scheduler.alphas, t, xt).sqrt(), sigma_t_square.sqrt())
        # x_t_prev = (x_t - eps_factor * eps_theta) / self.alphas[t].sqrt() + self.sigmas[t] * z
        #
        # sample_prev = x_t_prev
        # sample_prev = torch.cat((sample_prev, (pen_state > 0.5).float()), dim=2)
        ######################

        # Implement the DDIM reverse step
        t = t.to(x_t.device)
        alpha_t = self.alphas_cumprod[t]  # Shape: [B,1,1,1]
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()
        beta_t = self.betas[t]
        # Calculate x_0 estimate
        x_0 = (x_t - sqrt_one_minus_alpha_t * eps_theta) / sqrt_alpha_t

        # Handle t-1 (previous timestep)
        t_prev = t - self.num_train_timesteps // self.num_inference_timesteps
        if t_prev < 0:
            t_prev = 0
        alpha_t_prev = self.alphas_cumprod[t_prev]
        sqrt_alpha_t_prev = alpha_t_prev.sqrt()
        sqrt_one_minus_alpha_t_prev = (1 - alpha_t_prev).sqrt()

        eta = 0.5

        # Compute sigma for stochasticity
        sigma_t = eta * ((1 - alpha_t_prev) / (1 - alpha_t) * beta_t).sqrt()  # [B,1,1,1]
        # Compute the directional mean
        pred_mean = sqrt_alpha_t_prev * x_0 + sqrt_one_minus_alpha_t_prev * eps_theta * (1 - eta)

        z = torch.randn_like(x_t)
        x_t_prev = pred_mean + sigma_t * z

        sample_prev = x_t_prev
        # print(sample_prev.size(), (pen_state > 0.5).float().size())
        sample_prev = torch.cat((sample_prev, (pen_state > 0.5).float()), dim=2)
        
        return sample_prev
    
    # https://nn.labml.ai/diffusion/ddpm/utils.html
    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor): # get t th const 
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.IntTensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0).

        Input:
            x_0 (`torch.Tensor [B,C,H,W]`): samples from a real data distribution q(x_0).
            t: (`torch.IntTensor [B]`)
            eps: (`torch.Tensor [B,C,H,W]`, optional): if None, randomly sample Gaussian noise in the function.
        Output:
            x_t: (`torch.Tensor [B,C,H,W]`): noisy samples at timestep t.
            eps: (`torch.Tensor [B,C,H,W]`): injected noise.
        """
        
        if eps is None:
            eps       = torch.randn(x_0.shape, device='cuda')

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM forward step.
        alphas_prod_t = self.alphas_cumprod[t]
        x_t = torch.sqrt(alphas_prod_t) * x_0 + torch.sqrt(1 - alphas_prod_t) * eps
        #######################

        return x_t, eps
