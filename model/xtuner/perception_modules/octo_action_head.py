import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Same as your original: cosine schedule -> betas[t]
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, device=device, dtype=dtype) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0, 0.999)


class SimpleDiffusionXYActionHeadTorch(nn.Module):
    """
    Simplified version of your OctoDiffusionActionHeadTorch:
      - No window dimension W
      - No token pooling: you pass obs_embed directly
      - No masks / masked_mean
      - Action is (x,y) per sample: shape (B, 2) by default (action_horizon fixed to 1)
      - Diffusion math + schedule + network architecture kept the same style
    """

    def __init__(
        self,
        embed_dim: int,               # D
        action_dim: int = 2,          # (x, y)
        max_action: float = 5.0,
        loss_type: str = "mse",       # "mse" or "l1"
        time_dim: int = 32,
        num_blocks: int = 3,
        hidden_dim: int = 256,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = True,
        diffusion_steps: int = 20,
        n_diffusion_samples: int = 1,
    ):
        super().__init__()
        assert loss_type in ("mse", "l1")
        assert time_dim % 2 == 0

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.aflat = action_dim  # horizon=1 => flat dim is just action_dim
        self.max_action = max_action
        self.loss_type = loss_type

        self.time_dim = time_dim
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.diffusion_steps = diffusion_steps
        self.n_diffusion_samples = n_diffusion_samples

        # ---- time_preprocess: learnable FourierFeatures
        self.time_kernel = nn.Parameter(torch.empty(time_dim // 2, 1))
        nn.init.normal_(self.time_kernel, mean=0.0, std=0.2)

        # ---- cond_encoder MLP: 32 -> 64 -> 32
        self.cond_fc1 = nn.Linear(time_dim, 2 * time_dim)
        self.cond_fc2 = nn.Linear(2 * time_dim, time_dim)
        self._init_xavier_uniform(self.cond_fc1)
        self._init_xavier_uniform(self.cond_fc2)

        # ---- reverse_network: input [cond(32), obs(D), action(aflat)]
        reverse_in_dim = time_dim + embed_dim + self.aflat
        self.in_proj = nn.Linear(reverse_in_dim, hidden_dim)
        self._init_xavier_uniform(self.in_proj)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(self._make_res_block(hidden_dim, use_layer_norm=use_layer_norm, dropout_rate=dropout_rate))

        self.out_proj = nn.Linear(hidden_dim, self.aflat)
        self._init_xavier_uniform(self.out_proj)

        # ---- diffusion buffers (cosine)
        betas = cosine_beta_schedule(diffusion_steps)
        alphas = 1.0 - betas
        alpha_hats = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hats", alpha_hats)

        self.max_train_t = 5 # debug

    # ---------------- init helpers ----------------
    @staticmethod
    def _init_xavier_uniform(layer: nn.Linear) -> None:
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _init_lecun_normal(layer: nn.Linear) -> None:
        fan_in = layer.weight.shape[1]
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(layer.weight, mean=0.0, std=std)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _make_res_block(self, features: int, use_layer_norm: bool, dropout_rate: float):
        class ResBlock(nn.Module):
            def __init__(self, outer):
                super().__init__()
                self.dropout_rate = outer.dropout_rate
                self.use_layer_norm = outer.use_layer_norm
                self.ln = nn.LayerNorm(features) if self.use_layer_norm else None
                self.fc1 = nn.Linear(features, features * 4)
                self.fc2 = nn.Linear(features * 4, features)
                outer._init_lecun_normal(self.fc1)
                outer._init_lecun_normal(self.fc2)

            def forward(self, x: torch.Tensor, train: bool) -> torch.Tensor:
                residual = x
                if self.dropout_rate is not None and self.dropout_rate > 0.0:
                    x = F.dropout(x, p=self.dropout_rate, training=train)
                if self.ln is not None:
                    x = self.ln(x)
                x = self.fc1(x)
                x = F.silu(x)
                x = self.fc2(x)
                return residual + x

        return ResBlock(self)

    # ---------------- core pieces ----------------
    def fourier_features(self, time: torch.Tensor) -> torch.Tensor:
        """
        time: (..., 1)
        returns: (..., time_dim)
        """
        f = 2.0 * math.pi * (time @ self.time_kernel.t())
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)

    def cond_encoder(self, t_ff: torch.Tensor) -> torch.Tensor:
        """
        (..., 32) -> (..., 64) -> swish -> (..., 32)
        """
        x = self.cond_fc1(t_ff)
        x = F.silu(x)
        x = self.cond_fc2(x)
        return x

    def eps_pred(self, obs_embed: torch.Tensor, noisy_actions: torch.Tensor, time: torch.Tensor, train: bool) -> torch.Tensor:
        """
        obs_embed:     (B, D)
        noisy_actions: (..., B, aflat)
        time:          (..., B, 1)  float
        returns:       (..., B, aflat)
        """
        assert obs_embed.ndim == 2, "Expected obs_embed of shape (B, D)"
        B, D = obs_embed.shape
        assert D == self.embed_dim

        # time embedding
        t_ff = self.fourier_features(time)     # (...,B,32)
        cond = self.cond_encoder(t_ff)         # (...,B,32)

        # broadcast obs to match leading sample dims if present
        # (B,D) -> (...,B,D)
        if obs_embed.shape[:-1] != cond.shape[:-1]:
            obs = obs_embed.expand(cond.shape[:-1] + (D,))
        else:
            obs = obs_embed

        # reverse_input: (...,B, 32 + D + aflat)
        reverse_input = torch.cat([cond, obs, noisy_actions], dim=-1)

        x = self.in_proj(reverse_input)
        for blk in self.blocks:
            x = blk(x, train=train)
        x = F.silu(x)
        x = self.out_proj(x)
        return x

    # ---------------- training loss ----------------
    def loss(
        self,
        obs_embed: torch.Tensor,                 # (B, D)
        actions_xy: torch.Tensor,                # (B, 2) (or (B, action_dim))
        train: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Same diffusion objective as your original:
          sample t, add noise to x0, predict eps, regress to true eps (MSE/L1).
        No masks.
        """
        assert obs_embed.ndim == 2
        B, D = obs_embed.shape
        assert actions_xy.shape == (B, self.action_dim)

        # x0 clipped
        x0 = actions_xy.clamp(-self.max_action, self.max_action)  # (B,2)

        # Sample diffusion time: (n_samples, B, 1)
        # time_int = torch.randint(
        #     low=0, high=self.diffusion_steps,
        #     size=(self.n_diffusion_samples, B, 1),
        #     device=x0.device,
        #     generator=generator,
        #     dtype=torch.long,
        # )
        # debug: only train timesteps up to max_train_t
        max_train_t = min(self.diffusion_steps - 1, self.max_train_t)  # you define self.max_train_t
        # time_int = torch.randint(
        #     low=1, high=max_train_t + 1,  # [1, max_train_t]
        #     size=(self.n_diffusion_samples, B, 1),
        #     device=x0.device,
        #     generator=generator,
        #     dtype=torch.long,
        # )
        tt = torch.randint(1, max_train_t + 1, (1,), device=x0.device, generator=generator).item()
        time_int = torch.full((self.n_diffusion_samples, B, 1), tt, device=x0.device, dtype=torch.long)

        temp_g = torch.Generator(device=x0.device).manual_seed(tt)
        # Sample noise: (n_samples, B, aflat)
        noise = torch.randn((self.n_diffusion_samples, B, self.aflat), device=x0.device, generator=temp_g, dtype=torch.float32)
        # debug
        print(f"Training step with t={tt}, noise is:\n{noise}")

        # Forward marginal: x_t = sqrt(alpha_hat)*x0 + sqrt(1-alpha_hat)*eps
        alpha_hat_t = self.alpha_hats[time_int]  # (n_samples,B,1)
        scale = torch.sqrt(alpha_hat_t)
        std = torch.sqrt(1.0 - alpha_hat_t)
        noisy_actions = scale * x0.unsqueeze(0) + std * noise  # (n_samples,B,aflat)

        # Predict eps
        time_float = time_int.to(dtype=x0.dtype)  # (n_samples,B,1) 
        pred_eps = self.eps_pred(obs_embed, noisy_actions, time_float, train=False) # debug force it false  # (n_samples,B,aflat)

        # Loss
        if self.loss_type == "mse":
            loss_elem = (pred_eps - noise) ** 2
        else:
            loss_elem = (pred_eps - noise).abs()

        loss = loss_elem.mean()
        mse = ((pred_eps - noise) ** 2).mean()

        # Keep Octo-style scaling convention (optional but aligns with your original intent)
        loss = loss * self.action_dim
        mse = mse * self.action_dim

        metrics = {"loss": loss.detach(), "mse": mse.detach()}
        return loss, metrics

    # ---------------- sampling ----------------
    @torch.no_grad()
    def predict_action(
        self,
        obs_embed: torch.Tensor,                 # (B, D)
        train: bool = False,
        sample_shape: Tuple[int, ...] = (),
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Returns:
          action_xy: (*sample_shape, B, action_dim)
        Same DDPM ancestral sampler math as your original (minus masks / W / H).
        """
        device = device or obs_embed.device
        dtype = dtype or obs_embed.dtype

        assert obs_embed.ndim == 2
        B, D = obs_embed.shape
        assert D == self.embed_dim

        # init noise: (..., B, aflat)
        current_x = torch.randn((*sample_shape, B, self.aflat), device=device, generator=generator, dtype=dtype)

        debug_save = {'raw': current_x.detach().cpu()} # debug

        for t in range(self.diffusion_steps - 1, -1, -1):
            time_float = torch.full((*current_x.shape[:-1], 1), float(t), device=device, dtype=dtype)

            eps = self.eps_pred(obs_embed, current_x, time_float, train=train)

            alpha_1 = 1.0 / torch.sqrt(self.alphas[t])
            alpha_2 = (1.0 - self.alphas[t]) / torch.sqrt(1.0 - self.alpha_hats[t])
            current_x = alpha_1 * (current_x - alpha_2 * eps)

            # debug to disable it
            # z = torch.randn(current_x.shape, device=current_x.device, dtype=current_x.dtype, generator=generator)
            # if t > 0:
            #     current_x = current_x + torch.sqrt(self.betas[t]) * z

            debug_save[f'step_{t}'] = current_x.detach().cpu()
            current_x = current_x.clamp(-self.max_action, self.max_action)
            debug_save[f"alpha_1_{t}"] = alpha_1.detach().cpu()
            debug_save[f"alpha_2_{t}"] = alpha_2.detach().cpu()
            debug_save[f'step_{t}_clamped'] = current_x.detach().cpu()
            debug_save[f'eps_{t}'] = eps.detach().cpu()

        # output shape: (..., B, action_dim)
        return current_x, debug_save

    @torch.no_grad()
    def predict_action_from_gt_debug(
        self,
        obs_embed: torch.Tensor,                 # (B, D)
        actions_xy_gt: torch.Tensor,             # (B, action_dim)  <-- GT action
        train: bool = False,
        sample_shape: Tuple[int, ...] = (),
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        HARD-CODED DEBUG INFERENCE:
        1) take GT x0
        2) forward-noise it to x_{start_t}
        3) run reverse steps start_t -> 0 (deterministic, no z)
        This tests whether your model learned denoising at all.
        """
        device = device or obs_embed.device
        dtype = dtype or obs_embed.dtype

        assert obs_embed.ndim == 2
        B, D = obs_embed.shape
        assert actions_xy_gt.shape == (B, self.action_dim)

        T = self.diffusion_steps
        start_t = int(self.max_train_t)  # debug: start from max_train_t which is the highest t we train on
        start_t = max(0, min(start_t, T - 1))

        # x0 (GT) clipped
        x0 = actions_xy_gt.to(device=device, dtype=dtype).clamp(-self.max_action, self.max_action)

        # forward noise to x_{start_t}
        # noise shape: (*sample_shape, B, aflat)
        eps0 = torch.randn((*sample_shape, B, self.aflat), device=device, dtype=dtype, generator=generator)

        a_hat = self.alpha_hats[start_t].to(device=device, dtype=dtype)  # scalar tensor
        scale = torch.sqrt(a_hat)
        std = torch.sqrt(1.0 - a_hat)

        # current_x = x_{start_t}
        current_x = scale * x0.view((1,) * len(sample_shape) + (B, self.aflat)) + std * eps0

        debug_save = {
            "start_t": start_t,
            "x0_gt": x0.detach().cpu(),
            "eps0": eps0.detach().cpu(),
            "x_start": current_x.detach().cpu(),
            "alpha_hat_start": a_hat.detach().cpu(),
        }

        # reverse: start_t -> 0 (deterministic: NO z injection)
        for t in range(start_t, -1, -1):
            time_float = torch.full((*current_x.shape[:-1], 1), float(t), device=device, dtype=dtype)
            eps_pred = self.eps_pred(obs_embed, current_x, time_float, train=train)

            alpha_1 = 1.0 / torch.sqrt(self.alphas[t].to(device=device, dtype=dtype))
            alpha_2 = (1.0 - self.alphas[t].to(device=device, dtype=dtype)) / torch.sqrt(
                1.0 - self.alpha_hats[t].to(device=device, dtype=dtype)
            )
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            debug_save[f"step_{t}"] = current_x.detach().cpu()
            debug_save[f"eps_pred_{t}"] = eps_pred.detach().cpu()
            debug_save[f"alpha_1_{t}"] = alpha_1.detach().cpu()
            debug_save[f"alpha_2_{t}"] = alpha_2.detach().cpu()

        # final clamp once
        current_x = current_x.clamp(-self.max_action, self.max_action) 
        debug_save["final"] = current_x.detach().cpu()

        return current_x, debug_save
