import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import WeightedL1, WeightedL2

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        # Here the position encoding is the same as transformer
        # PE(pos, 2i) = sin(pos / 10000 ^ {2i / dmodel})
        # PE(pos, 2i + 1) = cos(pos / 10000 ^ {2i / dmodel})
        # consider the frequency part in the above equation: 1/10000^{2i / d_{model}}
        # = 10000 ^ {-2i / d_{model}} = exp (-2i / d * log(10000)) = exp(-i / (d/2) * log (10000))
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # here means two broadcasting: x [seq_len, 1], emb [1, half_dim] => [seq_len, half_dim]
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DenoiseNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, time_dim=16):
        super(DenoiseNetwork, self).__init__()
        
        self.time_dim = time_dim
        self.action_dim = action_dim
        
        # encode the time
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim * 2),
            nn.Mish(),  # commonly used in transformer architecture
            nn.Linear(self.time_dim * 2, time_dim)
        )
        
        input_dim = state_dim + action_dim + time_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        
        self.final_layer = nn.Linear(hidden_dim, action_dim)
        
        # neural network initialization
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, time, state):
        # here x menas action
        time_emb = self.time_mlp(time)
        x = torch.cat((x, state, time_emb), dim=1)
        x = self.mid_layer(x)
        return self.final_layer(x)
    
Losses = {
    "l1": WeightedL1,
    "l2": WeightedL2
}    

def extract(a, t, x_shape):
    # a denotes the register buffer
    # used to extract the data of timestep t
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1, ) * (len(x_shape) - 1)))

class DiffusionPolicy(nn.Module):
    def __init__(self, loss_type, beta_schedule="linear", clip_denoised=True, predict_epsilon=True, **kwargs):
        super(DiffusionPolicy, self).__init__()
        self.state_dim = kwargs["obs_dim"]
        self.action_dim = kwargs["act_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.T = kwargs["T"]    # the steps to denoise
        self.device = torch.device(kwargs["device"])
        
        # noise network
        # noise network is used to predict the initial noise to get x_t from x_0
        self.model = DenoiseNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        
        # discrete the time
        if beta_schedule == "linear":
            betas = torch.linspace(1e-4, 2e-2, self.T, dtype=torch.float).to(self.device)   # correspond to \beta_t
        alphas = 1 - betas  # correspond to \alpha_t
        
        # compute \overline{\alpha}_t, multiple step by step
        # alphas = [2, 3, 5]
        # alphas_cumprod = [2, 6, 30]
        alphas_cumprod = torch.cumprod(alphas, axis=0)  
        
        # compute \overline{\alpha}_{t-1} for posterior sampling
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(self.device), alphas_cumprod[:-1]])
        
        # register to buffer
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # forward process (noise)
        # x_t = \sqrt{1-\beta_t} x_{t-1} + \beta_t \epsilon = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon 
        # = \sqrt{\overline{\alpha}_t}x_0 + \sqrt{1-\overline{\alpha}_t}\epsilon
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1 - alphas_cumprod))
        
        # backward process (denoise)
        # q(x_{t-1}|x_t,x_0) -> N(x_{t-1}; \frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_t + \sqrt{\overline{\alpha_{t-1}}}(1-\alpha_t)x_0}{1-\overline{\alpha}_t}), 
        # \frac{(1-\alpha_t)(1-\overline{\alpha_{t-1}})}{1-\overline{\alpha}_t}}I
        posterior_variance = (
            betas * (1. - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clip", torch.log(posterior_variance).clamp(min=1e-20))
        
        # we need to estimate x_0, use x_t to denote
        # x_0 = (x_t - \sqrt{1-\overline{\alpha_t}}\epsilon_0 ) / \sqrt{\overline{\alpha_t}}
        self.register_buffer("sqrt_one_divide_alpha_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod_divide_alpha_cumprod", torch.sqrt((1. - alphas_cumprod) / alphas_cumprod))
        
        # register the coefficient to compute mean
        self.register_buffer("posterior_mean_coef_x0", betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef_xt", (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod))
        
        self.loss_fn = Losses[loss_type]()
    
    def q_posterior(self, x0, x, timestep):
        posterior_mean = (
            extract(self.posterior_mean_coef_x0, timestep, x.shape) * x0 
            + extract(self.posterior_mean_coef_xt, timestep, x.shape) * x
        )
        posterior_variance = extract(self.posterior_variance, timestep, x.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clip, timestep, x.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance
        
        
    def predict_state_from_noise(self, x, timestep, pred_noise):
        # x_0 = (x_t - \sqrt{1-\overline{\alpha_t}}\epsilon_0 ) / \sqrt{\overline{\alpha_t}}
        return (extract(self.sqrt_one_divide_alpha_cumprod, timestep, x.shape) * x
                - extract(self.sqrt_one_minus_alpha_cumprod_divide_alpha_cumprod, timestep, x.shape) * pred_noise)
        
    def p_mean_variance(self, x, timestep, state):
        # get the mean and variance of q(x_{t-1}|x_t,x_0)
        pred_noise = self.model(x, timestep, state)
        
        x_0_recon = self.predict_state_from_noise(x, timestep, pred_noise)
        x_0_recon.clamp_(-1, 1)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_0_recon, x, timestep)
        # log would be more stable
        return model_mean, posterior_log_variance
        
        
    def p_sample(self, x, timestep, state):
        # some noise (attach to standard normal distribution) should be put on the sampling variance
        model_mean, model_log_variance = self.p_mean_variance(x, timestep, state)
        noise = torch.randn_like(x).to(self.device)
        
        # the final step do not apply noise
        # assume x [a, b, c, d]
        # (1, ) * (len(x.shape) - 1) could generate [1, 1, 1]
        nonzero_mask = (~(timestep == 0)).float().reshape(x.shape[0], *((1, ) * (len(x.shape) - 1)))
        return model_mean + (0.5 * model_log_variance).exp() * nonzero_mask * noise
        
    
    def p_sample_loop(self, state, shape, *args, **kwargs):
        batch_size = state.shape[0]
        
        # generate the noise
        # here! require_grads True/False should tailor for specific cases
        x = torch.randn(shape, requires_grad=False).to(self.device)
        
        for i in reversed(range(0, self.T)):
            timestep = torch.full((batch_size, ), fill_value=i, dtype=torch.long).to(self.device)
            x = self.p_sample(x, timestep, state)
        return x
        
    def sample(self, state, *args, **kwargs):
        # state: [batch_size, state_dim]
        batch_size = state.shape[0]
        
        shape = [batch_size, self.action_dim]   # used to initialize the noise
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp(-1, 1)
    
    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)
    
    def train_label(self, x_start, timestep, noise):
        sample = (
            extract(self.sqrt_alphas_cumprod, timestep, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alpha_cumprod, timestep, x_start.shape) * noise
        )
        return sample
    
    def p_losses(self, x_start, state, timestep, weights):
        noise = torch.randn_like(x_start).to(self.device)
        xt = self.train_label(x_start, timestep, noise)
        noise_recon = self.model(xt, timestep, state)
        
        loss = self.loss_fn(noise_recon, noise, weights)
        return loss
    
    def loss(self, x, state, weights=1.0):
        # \epsilon - \epsilon_\theta(\sqrt{\overline{\alpha_t}}x_0 + \sqrt{1-\overline{\alpha_t}}\epsilon),t)
        # x denote the sample in the dataset, means x_0
        # in RL, x means action
        batch_size = x.shape[0]
        timestep = torch.randint(0, self.T, (batch_size, ), dtype=torch.long).to(self.device)
        return self.p_losses(x, state, timestep, weights)


if __name__ == '__main__':
    device = "cuda:0"
    x = torch.randn(256, 2).to(device)
    state = torch.randn(256, 11).to(device)
    
    model = DiffusionPolicy(loss_type="l2", obs_dim=11, act_dim=2, hidden_dim=256, T=100, device=device)
    action = model(state)
    for _ in range(1000000):
        loss = model.loss(x, state)
    
    print(action, loss)