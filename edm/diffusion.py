import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl


def extract(a: torch.Tensor,
            t: torch.Tensor,
            x_shape) -> torch.Tensor:
    """Get coefficients at given timesteps and reshape to [batch_size, 1, ...]."""
    bs, = t.shape
    assert x_shape[0] == bs

    a = a.to(t.device)
    out = a[t]

    assert out.shape[0] == bs

    return out.reshape([bs] + (len(x_shape) - 1) * [1])


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float64)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


# Simple diffusion model training
class Diffusion(pl.LightningModule):
  """Basic Diffusion Model."""

  def __init__(self, config, net, dim):
    super().__init__()
    # assert var_type in ('beta_forward', 'beta_reverse', 'learned')
    self.net = net
    self._dim = dim

    self._var_type = config.var_type
    self._n_steps = config.n_steps
    self._loss_type = config.loss_type
    self._mc_loss = config.mc_loss
    self._samples_per_step = config.samples_per_step
    self._betas = torch.from_numpy(cosine_beta_schedule(config.n_steps))
#    self._betas = torch.from_numpy(cosine_beta_schedule(config.n_steps)).to(self.device)

    self._alphas = 1. - self._betas
    self._log_alphas = torch.log(self._alphas)

    alphas = 1. - self._betas

    self._sqrt_alphas = torch.Tensor(torch.sqrt(alphas))
    self._sqrt_recip_alphas = torch.Tensor(1. / torch.sqrt(alphas))

    self._alphas_cumprod = torch.cumprod(self._alphas, axis=0)
    self._alphas_cumprod_prev = torch.cat([torch.Tensor([1]), self._alphas_cumprod[:-1]])
#    self._alphas_cumprod_prev = torch.cat([torch.Tensor([1]).to(self.device), self._alphas_cumprod[:-1]])
    self._sqrt_alphas_cumprod = torch.sqrt(self._alphas_cumprod)
    self._sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self._alphas_cumprod)
    self._log_one_minus_alphas_cumprod = torch.log(1 - self._alphas_cumprod)

    self._sqrt_recip_alphas_cumprod = 1 / torch.sqrt(self._alphas_cumprod)
    self._sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self._alphas_cumprod - 1)
    self._sqrt_recipm1_alphas_cumprod_custom = torch.sqrt(1. / (1 - self._alphas_cumprod))

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    self._posterior_variance = self._betas * (
        1. - self._alphas_cumprod_prev) / (1. - self._alphas_cumprod)

    self._posterior_log_variance_clipped = torch.log(
        torch.clip(self._posterior_variance, min=torch.min(self._betas), max=None))
    self._posterior_mean_coef1 = self._betas * torch.sqrt(
        self._alphas_cumprod_prev) / (1 - self._alphas_cumprod)
    self._posterior_mean_coef2 = (1 - self._alphas_cumprod_prev) * torch.sqrt(
        self._alphas) / (1 - self._alphas_cumprod)

#    self.device = device

  def energy_scale(self, t):
    return self._sqrt_recipm1_alphas_cumprod[t]

  def data_scale(self, t):
    return self._sqrt_recip_alphas_cumprod[t]

#  def forward(self, x, t):
#    """Get mu_t-1 given x_t."""
#    x = torch.atleast_2d(x)
#    t = torch.atleast_1d(t)
#
#
#    outs = self.net(x, t)
#    return outs

  def stats(self):
    """Returns static variables for computing variances."""
    return {
        'betas': self._betas,
        'alphas': self._alphas,
        'alphas_cumprod': self._alphas_cumprod,
        'alphas_cumprod_prev': self._alphas_cumprod_prev,
        'sqrt_alphas_cumprod': self._sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': self._sqrt_one_minus_alphas_cumprod,
        'log_one_minus_alphas_cumprod': self._log_one_minus_alphas_cumprod,
        'sqrt_recip_alphas_cumprod': self._sqrt_recip_alphas_cumprod,
        'sqrt_recipm1_alphas_cumprod': self._sqrt_recipm1_alphas_cumprod,
        'posterior_variance': self._posterior_variance,
        'posterior_log_variace_clipped': self._posterior_log_variance_clipped
    }

  def q_mean_variance(self, x_0, t):
    """Returns parameters of q(x_t | x_0)."""
    mean = extract(self._sqrt_alphas_cumprod, t, x_0.shape) * x_0
    variance = extract(1. - self._alphas_cumprod, t, x_0.shape)
    log_variance = extract(self._log_one_minus_alphas_cumprod, t, x_0.shape)
    return mean, variance, log_variance

  def q_sample(self, x_0, t, noise=None):
    """Sample from q(x_t | x_0)."""

    if noise is None:
      noise = np.random.normal(size=x_0.shape)

#    print("device in q_sample")
#    print(self.device)
#    x_t = extract(self._sqrt_alphas_cumprod, t, x_0.shape).to(self.device) * x_0 + extract(
#        self._sqrt_one_minus_alphas_cumprod, t, x_0.shape).to(self.device) * noise.to(self.device)

    x_t = extract(self._sqrt_alphas_cumprod, t, x_0.shape) * x_0 + extract(
        self._sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise

    return x_t

#   def p_loss_simple(self, x_0, t):
#     """Training loss for given x_0 and t."""
#     noise = torch.Tensor(np.random.normal(size=x_0.shape)).to(self.device)
#     x_noise = self.q_sample(x_0, t, noise)
#     noise_recon = self.forward(x_noise, t)
#     mse = torch.square(noise_recon - noise)
# 
#     mse = torch.mean(mse, axis=1)  # avg over the output dimension
# 
#     return mse
# 
#   def p_loss_kl(self, x_0, t):
#     """Training loss for given x_0 and t (KL-weighted)."""
# 
#     x_t = self.q_sample(x_0, t)
#     q_mean, _, q_log_variance = self.q_posterior(x_0, x_t, t)
#     p_mean, _, p_log_variance = self.p_mean_variance(x_t, t)
# 
#     dist_q = torch.distributions.Normal(q_mean, torch.exp(0.5 * q_log_variance))
#     def _loss(pmu, plogvar):
#       dist_p = torch.distributions.Normal(pmu, torch.exp(0.5 * plogvar))
#       kl = dist_q.kl_divergence(dist_p).mean(-1)
#       nll = -dist_p.log_prob(x_0).mean(-1)
#       return kl, nll, torch.where(t == 0, nll, kl)
# 
#     kl, nll, loss = _loss(p_mean, p_log_variance)
# 
#     return loss

  def q_posterior(self, x_0, x_t, t):
    """Obtain parameters of q(x_{t-1} | x_0, x_t)."""

    mean = (
        extract(self._posterior_mean_coef1, t, x_t.shape) * x_0
        + extract(self._posterior_mean_coef2, t, x_t.shape) * x_t
    )
    var = extract(self._posterior_variance, t, x_t.shape)
    log_var_clipped = extract(self._posterior_log_variance_clipped,
                              t, x_t.shape)
    return mean, var, log_var_clipped

  def predict_start_from_noise(self, x_t, t, noise):
    """Predict x_0 from x_t."""

    x_0 = (
        extract(self._sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
        - extract(self._sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )
    return x_0

  def forward(self, x, t, s):
    """Get mu_t-1 given x_t."""
    x = torch.atleast_2d(x)
    s = torch.atleast_2d(s)
    t = torch.atleast_1d(t)

    outs = self.net(x, t, s)
    return outs

  def p_loss_simple(self, x_0, t, s):
    """Training loss for given x_0 and t."""
    noise = torch.Tensor(np.random.normal(size=x_0.shape)).to(self.device)
    x_noise = self.q_sample(x_0, t, noise)
    noise_recon = self.forward(x_noise, t, s)
    mse = torch.square(noise_recon - noise)

    mse = torch.mean(mse, axis=1)  # avg over the output dimension

    return mse

  def p_loss_kl(self, x_0, t, s):
    """Training loss for given x_0 and t (KL-weighted)."""

    x_t = self.q_sample(x_0, t)
    q_mean, _, q_log_variance = self.q_posterior(x_0, x_t, t)
    p_mean, _, p_log_variance = self.p_mean_variance(x_t, t, s)

    dist_q = torch.distributions.Normal(q_mean, torch.exp(0.5 * q_log_variance))
    def _loss(pmu, plogvar):
      dist_p = torch.distributions.Normal(pmu, torch.exp(0.5 * plogvar))
      kl = dist_q.kl_divergence(dist_p).mean(-1)
      nll = -dist_p.log_prob(x_0).mean(-1)
      return kl, nll, torch.where(t == 0, nll, kl)

    kl, nll, loss = _loss(p_mean, p_log_variance)

    return loss

  def p_mean_variance(self, x, t, s, clip=torch.inf):
    """Parameters of p(x_{t-1} | x_t)."""

    x_recon = torch.clip(
        self.predict_start_from_noise(x, t, noise=self.forward(x, t, s)), -clip,
        clip)

    mean, var, log_var = self.q_posterior(x_recon, x, t)


    if self._var_type == 'beta_reverse':
      pass
    elif self._var_type == 'beta_forward':
      var = extract(self._betas, t, x.shape)
      log_var = torch.log(var)
    else:
      raise ValueError(f'{self._var_type} not recognised.')

    return mean, var, log_var

  def p_sample(self, x, t, s, rng_key=None, clip=torch.inf):
    """Sample from p(x_{t-1} | x_t)."""

    mean, _, log_var = self.p_mean_variance(x, t, s, clip=clip)

    noise = torch.Tensor(np.random.normal(size=x.shape)).to(self.device)

    x_tm1 = mean + torch.exp(0.5 * log_var) * noise
    return x_tm1

#  def _prior_kl(self, x_0):
#    """KL(q_T(x) || p(x))."""
#    t = torch.ones((x_0.shape[0],), dtype=torch.int64) * (self._n_steps - 1)
#    qt_mean, _, qt_log_variance = self.q_mean_variance(x_0, t)
#    qt_dist = torch.distributions.Normal(qt_mean, torch.exp(0.5 * qt_log_variance))
#    p_dist = torch.distributions.Normal(torch.zeros_like(qt_mean), torch.ones_like(qt_mean))
#    kl = torch.distributions.kl.kl_divergence(qt_dist, p_dist)
#    return kl

  def logpx(self, x_0, s):
    """Full elbo estimate of model."""
    e = self._prior_kl(x_0)
    n_repeats = self._n_steps * self._samples_per_step
    e = e.repeat(n_repeats, axis=0) / n_repeats

    kls = self.loss_all_t(x_0, s, loss_type='kl')
    logpx = -(kls + e) * self._dim * self._n_steps
    return {'logpx': logpx}


  def sample(self, s, n, clip=torch.inf):
    """Sample from p(x)."""

    x = torch.Tensor(np.random.normal(size=(n, self._dim))).to(self.device)
    s = s.unsqueeze(0).expand((n, -1))

    for i in range(self._n_steps):
      j = self._n_steps - 1 - i
      t = torch.ones((n,), dtype=torch.int64).to(self.device) * j
#      t = torch.ones((n,), dtype=torch.int64) * j
      x = self.p_sample(x, t, s, clip=clip)

    return x

  def loss(self, x, s):
    if self._mc_loss:
      return self.loss_mc(x, s, loss_type=self._loss_type)
    else:
      return self.loss_all_t(x, s, loss_type=self._loss_type)

  def loss_mc(self, x, s, loss_type=None):
    """Compute training loss, uniformly sampling t's."""

#    t = torch.from_numpy(np.random.randint(0, self._n_steps, size=(x.shape[0],))).to(self.device)
    t = torch.from_numpy(np.random.randint(0, self._n_steps, size=(x.shape[0],))).to(self.device)
    if loss_type == 'simple':
        loss = self.p_loss_simple(x, t, s)
    elif loss_type == 'kl':
        loss = self.p_loss_kl(x, t, s)
    else:
        raise ValueError(f'Unrecognized loss type: {loss_type}')

    return loss

  def loss_all_t(self, x, s, loss_type=None):
    """Compute training loss enumerated and averaged over all t's."""
    x = torch.Tensor(x).to(self.device)
    t = torch.concatenate([torch.arange(0, self._n_steps)] * x.shape[0])
    t = torch.tile(t[None], (self._samples_per_step,)).reshape(-1)
    x_r = torch.tile(x[None], (self._n_steps * self._samples_per_step,)).reshape(-1, *x.shape[1:])

    if loss_type == 'simple':
      loss = self.p_loss_simple(x_r, t, s)
    elif loss_type == 'kl':
      loss = self.p_loss_kl(x_r, t, s)
    else:
      raise ValueError(f'Unrecognized loss type: {loss_type}')
    return loss

  def p_gradient(self, x, t, s, clip=torch.inf):
    """Compute mean and variance of Gaussian reverse model p(x_{t-1} | x_t)."""
    b = x.shape[0]
    gradient = self.forward(x, t, s)
    gradient = gradient * extract(self._sqrt_recipm1_alphas_cumprod_custom, t, gradient.shape)

    return gradient

  def p_energy(self, x, t, s, clip=torch.inf):
    """Compute mean and variance of Gaussian reverse model p(x_{t-1} | x_t)."""
    b = x.shape[0]

    x = torch.atleast_2d(x)
    t = torch.atleast_1d(t)

    energy = self.net.neg_logp_unnorm(x, t, s)
    energy = energy * extract(self._sqrt_recipm1_alphas_cumprod_custom, t, energy.shape)

    return energy

#   def p_mean_variance(self, x, t, clip=torch.inf):
#     """Parameters of p(x_{t-1} | x_t)."""
# 
#     x_recon = torch.clip(
#         self.predict_start_from_noise(x, t, noise=self.forward(x, t)), -clip,
#         clip)
# 
#     mean, var, log_var = self.q_posterior(x_recon, x, t)
# 
# 
#     if self._var_type == 'beta_reverse':
#       pass
#     elif self._var_type == 'beta_forward':
#       var = extract(self._betas, t, x.shape)
#       log_var = torch.log(var)
#     else:
#       raise ValueError(f'{self._var_type} not recognised.')
# 
#     return mean, var, log_var
# 
#   def p_sample(self, x, t, rng_key=None, clip=torch.inf):
#     """Sample from p(x_{t-1} | x_t)."""
# 
#     mean, _, log_var = self.p_mean_variance(x, t, clip=clip)
# 
#     noise = torch.Tensor(np.random.normal(size=x.shape)).to(self.device)
# 
#     x_tm1 = mean + torch.exp(0.5 * log_var) * noise
#     return x_tm1
# 
#   def _prior_kl(self, x_0):
#     """KL(q_T(x) || p(x))."""
#     t = torch.ones((x_0.shape[0],), dtype=torch.int64) * (self._n_steps - 1)
#     qt_mean, _, qt_log_variance = self.q_mean_variance(x_0, t)
#     qt_dist = torch.distributions.Normal(qt_mean, torch.exp(0.5 * qt_log_variance))
#     p_dist = torch.distributions.Normal(torch.zeros_like(qt_mean), torch.ones_like(qt_mean))
#     kl = torch.distributions.kl.kl_divergence(qt_dist, p_dist)
#     return kl
# 
#   def logpx(self, x_0):
#     """Full elbo estimate of model."""
#     e = self._prior_kl(x_0)
#     n_repeats = self._n_steps * self._samples_per_step
#     e = e.repeat(n_repeats, axis=0) / n_repeats
# 
#     kls = self.loss_all_t(x_0, loss_type='kl')
#     logpx = -(kls + e) * self._dim * self._n_steps
#     return {'logpx': logpx}
# 
# 
#   def sample(self, n, clip=torch.inf):
#     """Sample from p(x)."""
# 
#     x = torch.Tensor(np.random.normal(size=(n, self._dim))).to(self.device)
# 
#     for i in range(self._n_steps):
#       j = self._n_steps - 1 - i
#       t = torch.ones((n,), dtype=torch.int64).to(self.device) * j
#       x = self.p_sample(x, t, clip=clip)
# 
#     return x
# 
#   def loss(self, x):
#     if self._mc_loss:
#       return self.loss_mc(x, loss_type=self._loss_type)
#     else:
#       return self.loss_all_t(x, loss_type=self._loss_type)
# 
#   def loss_mc(self, x, loss_type=None):
#     """Compute training loss, uniformly sampling t's."""
# 
#     t = torch.from_numpy(np.random.randint(0, self._n_steps, size=(x.shape[0],))).to(self.device)
#     if loss_type == 'simple':
#         loss = self.p_loss_simple(x, t)
#     elif loss_type == 'kl':
#         loss = self.p_loss_kl(x, t)
#     else:
#         raise ValueError(f'Unrecognized loss type: {loss_type}')
# 
#     return loss
# 
#   def loss_all_t(self, x, loss_type=None):
#     """Compute training loss enumerated and averaged over all t's."""
#     x = torch.Tensor(x).to(self.device)
#     t = torch.concatenate([torch.arange(0, self._n_steps)] * x.shape[0])
#     t = torch.tile(t[None], (self._samples_per_step,)).reshape(-1)
#     x_r = torch.tile(x[None], (self._n_steps * self._samples_per_step,)).reshape(-1, *x.shape[1:])
# 
#     if loss_type == 'simple':
#       loss = self.p_loss_simple(x_r, t)
#     elif loss_type == 'kl':
#       loss = self.p_loss_kl(x_r, t)
#     else:
#       raise ValueError(f'Unrecognized loss type: {loss_type}')
#     return loss
# 
#   def p_gradient(self, x, t, clip=torch.inf):
#     """Compute mean and variance of Gaussian reverse model p(x_{t-1} | x_t)."""
#     b = x.shape[0]
#     gradient = self.forward(x, t)
#     gradient = gradient * extract(self._sqrt_recipm1_alphas_cumprod_custom, t, gradient.shape)
# 
#     return gradient
# 
#   def p_energy(self, x, t, clip=torch.inf):
#     """Compute mean and variance of Gaussian reverse model p(x_{t-1} | x_t)."""
#     b = x.shape[0]
# 
#     x = torch.atleast_2d(x)
#     t = torch.atleast_1d(t)
# 
#     energy = self.net.neg_logp_unnorm(x, t)
#     energy = energy * extract(self._sqrt_recipm1_alphas_cumprod_custom, t, energy.shape)
# 
#     return energy
# 

# Simple diffusion model training
# class ConditionalDiffusionModel(DiffusionModel):
#   """Basic Diffusion Model."""
# 
#   def forward(self, x, t, s):
#     """Get mu_t-1 given x_t."""
#     x = torch.atleast_2d(x)
#     s = torch.atleast_2d(s)
#     t = torch.atleast_1d(t)
# 
#     outs = self.net(x, t, s)
#     return outs
# 
#   def p_loss_simple(self, x_0, t, s):
#     """Training loss for given x_0 and t."""
#     noise = torch.Tensor(np.random.normal(size=x_0.shape)).to(self.device)
#     x_noise = self.q_sample(x_0, t, noise)
#     noise_recon = self.forward(x_noise, t, s)
#     mse = torch.square(noise_recon - noise)
# 
#     mse = torch.mean(mse, axis=1)  # avg over the output dimension
# 
#     return mse
# 
#   def p_loss_kl(self, x_0, t, s):
#     """Training loss for given x_0 and t (KL-weighted)."""
# 
#     x_t = self.q_sample(x_0, t)
#     q_mean, _, q_log_variance = self.q_posterior(x_0, x_t, t)
#     p_mean, _, p_log_variance = self.p_mean_variance(x_t, t, s)
# 
#     dist_q = torch.distributions.Normal(q_mean, torch.exp(0.5 * q_log_variance))
#     def _loss(pmu, plogvar):
#       dist_p = torch.distributions.Normal(pmu, torch.exp(0.5 * plogvar))
#       kl = dist_q.kl_divergence(dist_p).mean(-1)
#       nll = -dist_p.log_prob(x_0).mean(-1)
#       return kl, nll, torch.where(t == 0, nll, kl)
# 
#     kl, nll, loss = _loss(p_mean, p_log_variance)
# 
#     return loss
# 
#   def p_mean_variance(self, x, t, s, clip=torch.inf):
#     """Parameters of p(x_{t-1} | x_t)."""
# 
#     x_recon = torch.clip(
#         self.predict_start_from_noise(x, t, noise=self.forward(x, t, s)), -clip,
#         clip)
# 
#     mean, var, log_var = self.q_posterior(x_recon, x, t)
# 
# 
#     if self._var_type == 'beta_reverse':
#       pass
#     elif self._var_type == 'beta_forward':
#       var = extract(self._betas, t, x.shape)
#       log_var = torch.log(var)
#     else:
#       raise ValueError(f'{self._var_type} not recognised.')
# 
#     return mean, var, log_var
# 
#   def p_sample(self, x, t, s, rng_key=None, clip=torch.inf):
#     """Sample from p(x_{t-1} | x_t)."""
# 
#     mean, _, log_var = self.p_mean_variance(x, t, s, clip=clip)
# 
#     noise = torch.Tensor(np.random.normal(size=x.shape)).to(self.device)
# 
#     x_tm1 = mean + torch.exp(0.5 * log_var) * noise
#     return x_tm1
# 
#   def _prior_kl(self, x_0):
#     """KL(q_T(x) || p(x))."""
#     t = torch.ones((x_0.shape[0],), dtype=torch.int64) * (self._n_steps - 1)
#     qt_mean, _, qt_log_variance = self.q_mean_variance(x_0, t)
#     qt_dist = torch.distributions.Normal(qt_mean, torch.exp(0.5 * qt_log_variance))
#     p_dist = torch.distributions.Normal(torch.zeros_like(qt_mean), torch.ones_like(qt_mean))
#     kl = torch.distributions.kl.kl_divergence(qt_dist, p_dist)
#     return kl
# 
#   def logpx(self, x_0, s):
#     """Full elbo estimate of model."""
#     e = self._prior_kl(x_0)
#     n_repeats = self._n_steps * self._samples_per_step
#     e = e.repeat(n_repeats, axis=0) / n_repeats
# 
#     kls = self.loss_all_t(x_0, s, loss_type='kl')
#     logpx = -(kls + e) * self._dim * self._n_steps
#     return {'logpx': logpx}
# 
# 
#   def sample(self, s, n, clip=torch.inf):
#     """Sample from p(x)."""
# 
#     x = torch.Tensor(np.random.normal(size=(n, self._dim))).to(self.device)
# 
#     for i in range(self._n_steps):
#       j = self._n_steps - 1 - i
#       t = torch.ones((n,), dtype=torch.int64).to(self.device) * j
#       x = self.p_sample(x, t, s, clip=clip)
# 
#     return x
# 
#   def loss(self, x, s):
#     if self._mc_loss:
#       return self.loss_mc(x, s, loss_type=self._loss_type)
#     else:
#       return self.loss_all_t(x, s, loss_type=self._loss_type)
# 
#   def loss_mc(self, x, s, loss_type=None):
#     """Compute training loss, uniformly sampling t's."""
# 
#     t = torch.from_numpy(np.random.randint(0, self._n_steps, size=(x.shape[0],))).to(self.device)
#     if loss_type == 'simple':
#         loss = self.p_loss_simple(x, t, s)
#     elif loss_type == 'kl':
#         loss = self.p_loss_kl(x, t, s)
#     else:
#         raise ValueError(f'Unrecognized loss type: {loss_type}')
# 
#     return loss
# 
#   def loss_all_t(self, x, s, loss_type=None):
#     """Compute training loss enumerated and averaged over all t's."""
#     x = torch.Tensor(x).to(self.device)
#     t = torch.concatenate([torch.arange(0, self._n_steps)] * x.shape[0])
#     t = torch.tile(t[None], (self._samples_per_step,)).reshape(-1)
#     x_r = torch.tile(x[None], (self._n_steps * self._samples_per_step,)).reshape(-1, *x.shape[1:])
# 
#     if loss_type == 'simple':
#       loss = self.p_loss_simple(x_r, t, s)
#     elif loss_type == 'kl':
#       loss = self.p_loss_kl(x_r, t, s)
#     else:
#       raise ValueError(f'Unrecognized loss type: {loss_type}')
#     return loss
# 
#   def p_gradient(self, x, t, s, clip=torch.inf):
#     """Compute mean and variance of Gaussian reverse model p(x_{t-1} | x_t)."""
#     b = x.shape[0]
#     gradient = self.forward(x, t, s)
#     gradient = gradient * extract(self._sqrt_recipm1_alphas_cumprod_custom, t, gradient.shape)
# 
#     return gradient
# 
#   def p_energy(self, x, t, s, clip=torch.inf):
#     """Compute mean and variance of Gaussian reverse model p(x_{t-1} | x_t)."""
#     b = x.shape[0]
# 
#     x = torch.atleast_2d(x)
#     t = torch.atleast_1d(t)
# 
#     energy = self.net.neg_logp_unnorm(x, t, s)
#     energy = energy * extract(self._sqrt_recipm1_alphas_cumprod_custom, t, energy.shape)
# 
#     return energy
