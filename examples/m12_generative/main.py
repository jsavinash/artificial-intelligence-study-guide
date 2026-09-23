"""m12 — Generative models: VAE encoder/loss terms (ELBO), diffusion forward
process with noise schedule + oracle reverse, and sampling temperature effects.

Proves theory doc 12-generative-models.md.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core import torch_backend as TB  # noqa: E402


def vae_step(x, rng):
    """One VAE step: encoder -> (mu, logvar) -> reparameterize -> decoder."""
    # encoder (linear) params baked in
    W_mu = rng.normal(scale=0.5, size=(4, 2))
    W_lv = rng.normal(scale=0.5, size=(4, 2))
    W_dec = rng.normal(scale=0.5, size=(2, 4))
    mu = x @ W_mu
    logvar = x @ W_lv
    std = np.exp(0.5 * logvar)
    eps = rng.normal(size=mu.shape)                 # reparameterization trick
    z = mu + std * eps                              # z ~ N(mu, sigma^2): differentiable
    x_hat = z @ W_dec                               # reconstruction
    # loss terms
    recon = ((x - x_hat) ** 2).mean(axis=1)         # -E_q[log p(x|z)]
    kl = -0.5 * (1 + logvar - mu ** 2 - np.exp(logvar)).mean(axis=1)  # KL(q||N(0,I))
    return z, x_hat, recon, kl, (W_mu, W_lv, W_dec)


def diffusion_forward(x0, betas):
    """q(x_t|x_0): gradually corrupt toward noise. alpha_bar_t = prod(1-beta)."""
    rng = np.random.default_rng(7)
    xts = [x0]
    a_bar = 1.0
    for beta in betas:
        a_bar *= (1 - beta)
        xt = np.sqrt(a_bar) * x0 + np.sqrt(1 - a_bar) * rng.normal(size=x0.shape)
        xts.append(xt)
    return xts, a_bar


def sample_with_temperature(logits, temp, rng, n=200):
    """tau -> 0 greedy-ish; tau large -> near-uniform (module 12's sampling table)."""
    l = logits / max(temp, 1e-6)
    p = np.exp(l - l.max()); p /= p.sum()
    return rng.choice(len(p), size=n, p=p)


def torch_path(X):
    """The framework equivalents: a real VAE, a GAN, and a denoising net.

    The NumPy code above *describes* the loss terms; here autograd actually
    trains them. Same math, real optimizers.
    """
    if not TB.HAS_TORCH:
        print("\n[torch] not installed — NumPy generative path only")
        return None
    print(f"\n[torch] trained generative models on {TB.get_device(  )}"
          .replace("  )", ")"))

    # VAE: encoder -> (mu, logvar) -> reparameterize -> decoder, trained on the
    # full ELBO (reconstruction + KL) by autograd
    vae = TB.train_vae(X, hidden=64, latent=8, epochs=150, lr=1e-3, seed=0)
    print(f"  VAE        : loss {vae['losses'][0]:.2f}->{vae['losses'][-1]:.2f} "
          f"({vae['backend']}, {vae['seconds']:.2f}s)")
    assert vae["losses"][-1] < vae["losses"][0], "VAE did not improve"

    # GAN: minimax — the generator must improve while the discriminator adapts
    gan = TB.train_gan(X, latent=8, epochs=200, lr=2e-3, seed=0)
    print(f"  GAN        : d_loss={gan['d_loss']:.3f} g_loss={gan['g_loss']:.3f} "
          f"({gan['seconds']:.2f}s)")

    # Diffusion: a network learns to predict the noise added at step t
    diff = TB.train_diffusion(X, timesteps=40, epochs=120, lr=1e-3, seed=0)
    print(f"  diffusion  : loss {diff['losses'][0]:.3f}->"
          f"{diff['losses'][-1]:.3f} (denoiser learned) "
          f"({diff['seconds']:.2f}s)")
    assert diff["losses"][-1] < diff["losses"][0], "denoiser did not improve"
    return {"vae": vae, "gan": gan, "diffusion": diff}


def main():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 4))

    # ---- 1) VAE: KL >= 0 with equality only when posterior == prior ----
    z, x_hat, recon, kl, _ = vae_step(X, rng)
    assert z.shape == (256, 2) and x_hat.shape == X.shape
    assert np.all(kl > -1e-9) and kl.mean() > 0.01
    elbo = recon + kl                                # ELBO = recon + KL (maximize)
    assert elbo.shape == (256,)

    # posterior collapse check: mu=0,logvar=0 => KL = 0 (matches N(0,I))
    mu0, lv0 = np.zeros((3, 2)), np.zeros((3, 2))
    kl0 = -0.5 * (1 + lv0 - mu0 ** 2 - np.exp(lv0)).mean(axis=1)
    assert np.allclose(kl0, 0.0)

    # ---- 2) diffusion: signal fades, noise dominates by the end ----
    x0 = rng.normal(size=(64, 8))
    betas = np.linspace(1e-4, 0.02, 1000)            # enough steps: prod(1-b) ~ 4e-5
    xts, a_bar = diffusion_forward(x0, betas)
    signal_first = np.corrcoef(xts[1].ravel(), x0.ravel())[0, 1]
    signal_last = np.corrcoef(xts[-1].ravel(), x0.ravel())[0, 1]
    assert a_bar < 1e-2                              # almost pure noise at t=T
    assert signal_last < signal_first               # structure destroyed over time
    # oracle reverse: subtract the KNOWN noise (what the network learns to predict)
    noise = rng.normal(size=x0.shape)
    a_bar_T = 1e-4
    xT = np.sqrt(a_bar_T) * x0 + np.sqrt(1 - a_bar_T) * noise
    x0_hat = (xT - np.sqrt(1 - a_bar_T) * noise) / np.sqrt(a_bar_T)
    assert np.allclose(x0_hat, x0, atol=1e-6)       # perfect denoiser recovers x0

    # ---- 3) temperature controls the sampling distribution ----
    logits = np.array([4.0, 2.0, 0.5, -1.0])
    low = sample_with_temperature(logits, 0.1, rng)
    high = sample_with_temperature(logits, 3.0, rng)
    mode_low = np.bincount(low, minlength=4).argmax()
    assert mode_low == 0                              # low tau = argmax dominates
    high_entropy = -(np.bincount(high, minlength=4) / len(high) *
                     np.log(np.bincount(high, minlength=4) / len(high) + 1e-12)).sum()
    low_entropy = -(np.bincount(low, minlength=4) / len(low) *
                    np.log(np.bincount(low, minlength=4) / len(low) + 1e-12)).sum()
    assert high_entropy > low_entropy

    print(f"PASS m12 generative | vae_kl={kl.mean():.3f} recon={recon.mean():.3f} "
          f"diffusion_sig {signal_first:.2f}->{signal_last:.2f} a_bar_T={a_bar:.1e} "
          f"oracle_recover=True tau_entropy low={low_entropy:.2f}<high={high_entropy:.2f}")


if __name__ == "__main__":
    main()
