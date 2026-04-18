"""
ppo_agent.py — PPO for Walker2D custom physics

obs_dim = 20  (17 Walker2D state + 3 horizontal target direction cues)
act_dim = 6   (6 continuous joint torques in [-1, 1])
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, math, threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from torchview import draw_graph


OBS_DIM = 20
ACT_DIM = 6

# ── Hyperparameters ───────────────────────────────────────────────────
LR_ACTOR           = 3e-4
LR_CRITIC          = 1e-3
GAMMA              = 0.99
GAE_LAM            = 0.95
CLIP_EPS           = 0.2
ENTROPY_COEF       = 0.01
VF_COEF            = 0.5
MAX_GRAD_NORM      = 0.5
N_EPOCHS           = 10
BATCH_SIZE         = 256
ROLLOUT_LEN        = 2048
LOG_STD_MIN        = -5.0
LOG_STD_MAX        = 2.0


# ── Network ───────────────────────────────────────────────────────────
def visualize_network(model, title="Network", save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn as nn

    layers = []
    weights = []

    # Extract structure
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            w = layer.weight.detach().cpu().numpy()
            weights.append(w)
            layers.append(w.shape[1])  # input size
    layers.append(weights[-1].shape[0])  # output size

    fig, ax = plt.subplots(figsize=(12, 7))

    # Layout parameters
    layer_spacing = 2.0
    neuron_spacing = 1.0

    positions = []

    # Draw neurons
    for i, layer_size in enumerate(layers):
        y_positions = np.linspace(
            -layer_size / 2, layer_size / 2, layer_size
        )
        x_positions = np.full(layer_size, i * layer_spacing)

        layer_pos = list(zip(x_positions, y_positions))
        positions.append(layer_pos)

        for (x, y) in layer_pos:
            circle = plt.Circle((x, y), 0.15, color='skyblue', ec='black')
            ax.add_patch(circle)

    # Draw connections
    for i, w in enumerate(weights):
        for j in range(w.shape[1]):   # input neuron
            for k in range(w.shape[0]):  # output neuron
                x1, y1 = positions[i][j]
                x2, y2 = positions[i + 1][k]

                weight = w[k, j]

                # Normalize weight for visibility
                alpha = min(1.0, abs(weight) / 1.0)

                color = 'green' if weight > 0 else 'red'

                ax.plot([x1, x2], [y1, y2],
                        color=color,
                        alpha=alpha * 0.3,
                        linewidth=0.5)

    ax.set_title(title)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"🧠 Network saved → {save_path}")

    plt.show()
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256)):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM):
        super().__init__()
        self.actor_mean    = MLP(obs_dim, act_dim)
        self.actor_log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.critic        = MLP(obs_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.net[-1].bias)

    def forward(self, x):
        mean    = self.actor_mean(x)
        log_std = self.actor_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        dist    = Normal(mean, log_std.exp().expand_as(mean))
        val     = self.critic(x).squeeze(-1)
        return dist, val

    def get_action(self, x, deterministic=False):
        dist, val = self(x)
        raw  = dist.mean if deterministic else dist.rsample()
        lp   = dist.log_prob(raw).sum(-1)
        act  = torch.tanh(raw)
        return act, lp, val

    def evaluate(self, x, action_raw):
        dist, val = self(x)
        lp      = dist.log_prob(action_raw).sum(-1)
        entropy = dist.entropy().sum(-1)
        return lp, entropy, val


# ── Rollout buffer ─────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, n_agents, obs_dim=OBS_DIM, act_dim=ACT_DIM, length=ROLLOUT_LEN):
        self.n   = n_agents
        self.T   = length
        self.od  = obs_dim
        self.ad  = act_dim
        self.clear()

    def clear(self):
        T, N = self.T, self.n
        self.obs   = np.zeros((T, N, self.od), np.float32)
        self.acts  = np.zeros((T, N, self.ad), np.float32)
        self.lps   = np.zeros((T, N),          np.float32)
        self.rews  = np.zeros((T, N),          np.float32)
        self.vals  = np.zeros((T, N),          np.float32)
        self.dones = np.zeros((T, N),          np.float32)
        self.ptr   = 0
        self.full  = False

    def add(self, obs, acts, lps, rews, vals, dones):
        t = self.ptr
        self.obs[t]   = obs
        self.acts[t]  = acts
        self.lps[t]   = lps
        self.rews[t]  = rews
        self.vals[t]  = vals
        self.dones[t] = dones
        self.ptr += 1
        if self.ptr >= self.T:
            self.full = True
            self.ptr  = 0

    def gae_returns(self, last_vals):
        T   = self.T
        adv = np.zeros((T, self.n), np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nv    = last_vals if t == T - 1 else self.vals[t + 1]
            delta = self.rews[t] + GAMMA * nv * (1 - self.dones[t]) - self.vals[t]
            gae   = delta + GAMMA * GAE_LAM * (1 - self.dones[t]) * gae
            adv[t] = gae
        return adv, adv + self.vals

    def batches(self, adv, ret):
        total  = self.T * self.n
        idx    = np.random.permutation(total)
        fo, fa, flp = (self.obs.reshape(total, self.od),
                       self.acts.reshape(total, self.ad),
                       self.lps.reshape(total))
        fadv   = adv.reshape(total)
        fret   = ret.reshape(total)
        for s in range(0, total, BATCH_SIZE):
            b = idx[s:s + BATCH_SIZE]
            yield fo[b], fa[b], flp[b], fadv[b], fret[b]


# ── Agent ──────────────────────────────────────────────────────────────

class PPOAgent:
    """
    PPO agent. Background training thread consumes the rollout buffer.

    Key design:
    • get_action_and_info()  — stochastic, used during training collection
    • get_best_action()      — DETERMINISTIC (mean of policy), used when
                               running the best loaded checkpoint for demo/eval
    """

    def __init__(self, target, load_existing=True, device='cpu', number_of_population=3):
        self.target               = np.asarray(target, dtype=np.float32)
        self.device               = torch.device(device)
        self.number_of_population = number_of_population

        self.lock        = threading.Lock()
        self.stop_flag   = threading.Event()
        self.train_thread = None

        self.episode_rewards            = []
        self.episode_reward_accumulator = np.zeros(number_of_population, np.float32)

        self.ac       = ActorCritic().to(self.device)   # training network
        self.ac_infer = ActorCritic().to(self.device)   # inference copy (always synced)
        self._sync_infer()

        self.opt_actor  = optim.Adam(
            list(self.ac.actor_mean.parameters()) + [self.ac.actor_log_std],
            lr=LR_ACTOR, eps=1e-5)
        self.opt_critic = optim.Adam(
            self.ac.critic.parameters(), lr=LR_CRITIC, eps=1e-5)

        self.sched_actor  = optim.lr_scheduler.StepLR(self.opt_actor,  200, 0.9)
        self.sched_critic = optim.lr_scheduler.StepLR(self.opt_critic, 200, 0.9)

        self.buffer      = RolloutBuffer(number_of_population)
        self.total_steps = 0
        self.is_training = False
        self._update_n   = 0
        self.best_reward = -float('inf')
        self.on_plot_updated = False

        # Whether to run deterministic (best-model demo) or stochastic (training)
        self._deterministic = False

        os.makedirs('checkpoints', exist_ok=True)
        if load_existing:
            if not self.load('checkpoints/walker_ppo_best.pt'):
                self.load('checkpoints/walker_ppo.pt')

    # ── Inference ─────────────────────────────────────────────────────
    def visualize_actor_critic(self):
        visualize_network(self.ac.actor_mean, "Actor Network", "actor_network.png")
        visualize_network(self.ac.critic, "Critic Network", "critic_network.png")
    @torch.no_grad()
    def get_action_and_info(self, obs: np.ndarray):
        """Stochastic action + log_prob + value — used during rollout collection."""
        t                  = torch.from_numpy(obs).to(self.device)
        act, lp, val       = self.ac.get_action(t, deterministic=False)
        return act.cpu().numpy(), lp.cpu().numpy(), val.cpu().numpy()

    @torch.no_grad()
    def get_best_action(self, obs: np.ndarray) -> np.ndarray:
        """DETERMINISTIC action from inference copy — best performance, no exploration noise."""
        t       = torch.from_numpy(obs).to(self.device)
        act, _, _ = self.ac_infer.get_action(t, deterministic=True)
        return act.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Unified entry point: deterministic if best-model mode, else stochastic."""
        if self._deterministic:
            return self.get_best_action(obs)
        t       = torch.from_numpy(obs).to(self.device)
        dist, _ = self.ac_infer(t)
        return torch.tanh(dist.rsample()).cpu().numpy().astype(np.float32)

    # ── Training control ──────────────────────────────────────────────

    def start_training(self):
        if self.is_training:
            return
        self._deterministic = False
        self.is_training    = True
        self.stop_flag.clear()
        self.train_thread = threading.Thread(
            target=self._train_loop, daemon=True, name='ppo_train')
        self.train_thread.start()
        print("🏋️  PPO training started")

    def stop_training(self):
        self.is_training = False
        self.stop_flag.set()
        if self.train_thread:
            self.train_thread.join(timeout=5)
        print("⏹  PPO training stopped")

    def enable_best_mode(self):
        """Switch to deterministic inference (no exploration noise)."""
        self._deterministic = True
        print("🏆  Best-model mode: deterministic inference ON")

    def disable_best_mode(self):
        self._deterministic = False
        print("🔀  Training mode: stochastic inference ON")

    # ── Checkpoint ────────────────────────────────────────────────────

    def save(self, path='checkpoints/walker_ppo.pt'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'ac':             self.ac.state_dict(),
            'opt_actor':      self.opt_actor.state_dict(),
            'opt_critic':     self.opt_critic.state_dict(),
            'sched_actor':    self.sched_actor.state_dict(),
            'sched_critic':   self.sched_critic.state_dict(),
            'total_steps':    self.total_steps,
            'best_reward':    self.best_reward,
            'update_count':   self._update_n,
            'episode_rewards': self.episode_rewards[-500:],
        }, path)
        print(f"💾 Saved → {path}  (steps={self.total_steps:,})")

    def load(self, path='checkpoints/walker_ppo.pt'):
        if not os.path.exists(path):
            print(f"ℹ️  No checkpoint at {path}")
            return False
        try:
            ck = torch.load(path, map_location=self.device)

            # Legacy key remap
            sd = ck.get('actor_critic', ck.get('ac', None))
            if sd is None:
                raise KeyError("No actor_critic / ac key in checkpoint")
            if 'actor_logstd' in sd and 'actor_log_std' not in sd:
                sd['actor_log_std'] = sd.pop('actor_logstd')
            if 'actor_log_standard_deviation' in sd and 'actor_log_std' not in sd:
                sd['actor_log_std'] = sd.pop('actor_log_standard_deviation')

            self.ac.load_state_dict(sd, strict=False)

            for key, opt in [('opt_actor',  self.opt_actor),
                             ('optimizer_actor', self.opt_actor),
                             ('opt_critic', self.opt_critic),
                             ('optimizer_critic', self.opt_critic)]:
                if key in ck:
                    try: opt.load_state_dict(ck[key])
                    except: pass
                    break

            self.total_steps = ck.get('total_steps', 0)
            self.best_reward = ck.get('best_reward',  -float('inf'))
            self._update_n   = ck.get('update_count', 0)
            if 'episode_rewards' in ck:
                self.episode_rewards = list(ck['episode_rewards'])

            self._sync_infer()
            print(f"📂 Loaded {path}  steps={self.total_steps:,}  "
                  f"best_reward={self.best_reward:.1f}")
            return True
        except Exception as e:
            print(f"❌ Load failed ({path}): {e}")
            import traceback; traceback.print_exc()
            return False

    def set_target(self, x, y, z):
        self.target[:] = [x, y, z]

    def close(self):
        self.stop_training()

    # ── Buffer recording ──────────────────────────────────────────────

    def record_step(self, obs, acts, lps, rews, vals, dones):
        self.buffer.add(obs, acts, lps, rews, vals, dones)
        self.total_steps += self.number_of_population
        self.episode_reward_accumulator += rews
        for i, d in enumerate(dones):
            if d:
                self.episode_rewards.append(float(self.episode_reward_accumulator[i]))
                self.episode_reward_accumulator[i] = 0.0

    # ── Training thread ───────────────────────────────────────────────

    def _train_loop(self):
        import time
        while not self.stop_flag.is_set():
            if self.buffer.full:
                self._update()
                self.buffer.clear()
            time.sleep(0.001)

    def _update(self):
        last_v        = np.zeros(self.number_of_population, np.float32)
        adv, ret      = self.buffer.gae_returns(last_v)
        adv           = (adv - adv.mean()) / (adv.std() + 1e-8)

        pl_sum = vl_sum = n = 0

        for _ in range(N_EPOCHS):
            for obs_b, act_b, lp_b, adv_b, ret_b in self.buffer.batches(adv, ret):
                ot  = torch.from_numpy(obs_b).to(self.device)
                at  = torch.from_numpy(act_b).to(self.device)
                lpt = torch.from_numpy(lp_b).to(self.device)
                at_ = torch.from_numpy(adv_b).to(self.device)
                rt  = torch.from_numpy(ret_b).to(self.device)

                ar  = torch.atanh(at.clamp(-0.9999, 0.9999))
                nlp, ent, val = self.ac.evaluate(ot, ar)

                ratio = (nlp - lpt).exp()
                s1    = ratio * at_
                s2    = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * at_
                pl    = -(torch.min(s1, s2) + ENTROPY_COEF * ent).mean()
                vl    = VF_COEF * (rt - val).pow(2).mean()

                self.opt_actor.zero_grad()
                pl.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(
                    list(self.ac.actor_mean.parameters()) + [self.ac.actor_log_std],
                    MAX_GRAD_NORM)
                self.opt_actor.step()

                self.opt_critic.zero_grad()
                vl.backward()
                nn.utils.clip_grad_norm_(
                    self.ac.critic.parameters(), MAX_GRAD_NORM)
                self.opt_critic.step()

                pl_sum += pl.item(); vl_sum += vl.item(); n += 1

        self.sched_actor.step()
        self.sched_critic.step()
        self._update_n += 1
        self._sync_infer()
        self.save()

        if self.episode_rewards:
            recent = float(np.mean(self.episode_rewards[-10:]))
            if recent > self.best_reward:
                self.best_reward = recent
                self.save('checkpoints/walker_ppo_best.pt')
                print(f"🏆 New best policy  mean_reward={recent:.1f}")

        self._plot_rewards()
        print(f"✅ Update #{self._update_n}  steps={self.total_steps:,}  "
              f"pl={pl_sum/max(n,1):.4f}  vl={vl_sum/max(n,1):.4f}  "
              f"lr={self.opt_actor.param_groups[0]['lr']:.2e}")

    def _sync_infer(self):
        with self.lock:
            self.ac_infer.load_state_dict(self.ac.state_dict())

    def _plot_rewards(self, path='checkpoints/reward_curve.png'):
        if len(self.episode_rewards) < 2:
            return
        try:
            tmp_path = path.replace('.png', '_tmp.png')
            r   = np.array(self.episode_rewards)
            ep  = np.arange(len(r))
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#0a0c10')
            ax.set_facecolor('#0f1318')
            ax.plot(ep, r, color='#1e3a4a', lw=0.8, alpha=0.6)
            w = max(1, len(r) // 20)
            if len(r) >= w:
                sm = np.convolve(r, np.ones(w) / w, mode='valid')
                ax.plot(np.arange(len(sm)) + w // 2, sm,
                        color='#00d4ff', lw=2.0, label=f'Smooth (w={w})')
            bi = int(np.argmax(r))
            ax.axhline(r[bi], color='#00ff88', lw=0.8, ls='--', alpha=0.5,
                       label=f'Best: {r[bi]:.1f}')
            ax.scatter([bi], [r[bi]], color='#00ff88', s=60, zorder=5)
            ax.set_xlabel('Episode', color='#4a5568', fontsize=9)
            ax.set_ylabel('Total Reward', color='#4a5568', fontsize=9)
            ax.set_title(f'Reward | steps={self.total_steps:,} | update #{self._update_n}',
                         color='#cbd5e1', fontsize=10)
            ax.tick_params(colors='#4a5568')
            for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
            for sp in ['bottom', 'left']: ax.spines[sp].set_color('#1c2333')
            ax.legend(fontsize=8, facecolor='#0f1318',
                      edgecolor='#1c2333', labelcolor='#cbd5e1')
            ax.grid(True, color='#1c2333', lw=0.5, alpha=0.7)
            fig.tight_layout()
            fig.savefig(tmp_path, format='png', dpi=120, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            os.replace(tmp_path, path)
            if callable(self.on_plot_updated):
                self.on_plot_updated()
            print(f"📊 Plot → {path}")
        except Exception as e:
            print(f"⚠️  Plot failed: {e}")
