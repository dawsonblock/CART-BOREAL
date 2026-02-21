import sys
import os
import numpy as np
import math

try:
    import gymnasium as gym
except ImportError:
    print("Please install gymnasium: pip install gymnasium")
    sys.exit(1)

# Import the existing sovereign engine
sys.path.append(os.path.expanduser("~/x1"))
from boreal_apex_sovereign_v2 import (
    ALU_Q16,
    ApexConfig,
    MetaEpistemicEnsemble,
    TriStateGateQ16,
    q16_cem_plan,
    ApexLogger,
    config,
)

# Monkey-patch config for CartPole physics
config.s_dim = 12
config.o_dim = 5
config.a_dim = 1  # CartPole only has 1 actuator (track track)
config.base_lr_shift = 7
config.recov_lr_shift = 4
config.cem_pop = 50
config.cem_iters = 3
config.horizon = 15


class CartPoleFixedPointWrapper(gym.Wrapper):
    """
    Wraps the CartPole environment to map continuous actions to Q16.
    """

    def __init__(self, env):
        super().__init__(env)
        # We replace the discontinuous theta with sin/cos components targeting 0
        self.obs_dim = 5

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # We manually structure the initial observation state
        x, x_dot, theta, theta_dot = self.env.unwrapped.state

        # Pad obs
        padded_obs = np.zeros(config.o_dim, dtype=np.float64)
        padded_obs[0] = x * 0.02
        padded_obs[1] = x_dot * 0.02
        padded_obs[2] = math.sin(theta) * 1.0
        padded_obs[3] = (1.0 - math.cos(theta)) * 1.0
        padded_obs[4] = theta_dot * 0.05

        return ALU_Q16.to_q(padded_obs), info

    def step(self, action_q16):
        # action_q16 is an array of size a_dim=1.
        # Convert fixed point to float
        action_f = ALU_Q16.to_f(action_q16[0])

        # BOREAL outputs continuous linear actions (e.g. -1.5, +0.3)
        # Gym expects a Discrete 0 (Left -10.0N) or 1 (Right +10.0N).
        # To bridge the purely linear cognitive topology of BOREAL with Gym,
        # we dynamically throttle the CartPole physical force magnitude!

        # Scale force linearly to the action. Clamped to realistic limits.
        force_n = np.clip(abs(action_f) * 10.0, 0.0, 20.0)
        self.env.unwrapped.force_mag = float(force_n)

        # Determine direction
        discrete_action = 1 if action_f > 0 else 0

        obs, reward, terminated, truncated, info = self.env.step(discrete_action)

        # Bounded Wall Physics (Continuous Non-Resetting Domain)
        x, x_dot, theta, theta_dot = self.env.unwrapped.state

        # Add physical walls at the edges of the track (-2.0 and +2.0)
        # If the cart hits the wall, mathematically bounce it back by reversing its velocity
        if x > 2.0:
            x = 2.0
            x_dot = -x_dot * 0.5  # Bounce and dampen velocity
        elif x < -2.0:
            x = -2.0
            x_dot = -x_dot * 0.5  # Bounce and dampen velocity

        # Wrap theta nicely between -pi and pi to stop internal gym termination bounds from triggering
        theta = ((theta + math.pi) % (2 * math.pi)) - math.pi

        # Inject wrapped physics back into gym
        self.env.unwrapped.state = (x, x_dot, theta, theta_dot)

        padded_obs = np.zeros(config.o_dim, dtype=np.float64)
        padded_obs[0] = x * 0.02
        padded_obs[1] = x_dot * 0.02
        padded_obs[2] = math.sin(theta) * 1.0
        padded_obs[3] = (1.0 - math.cos(theta)) * 1.0
        padded_obs[4] = theta_dot * 0.05

        # Calculate a continuous tracking reward based on how upright the pole is
        # We need the pole balancing UP (theta near 0).
        # If it drops past horizontal (-pi/2 or pi/2), mathematically punish the agent heavily.
        if abs(theta) > (math.pi / 2.0):
            upright_reward = -5.0  # Heavy penalty for hanging downside down
        else:
            # Reward scales from 0 (horizontal) up to 1 (perfectly vertical)
            upright_reward = math.cos(theta)

        # Extremely aggressive spin penalty so it stops doing circles
        spin_penalty = 1.0 * (theta_dot**2)
        center_penalty = 0.1 * abs(x)

        continuous_reward = upright_reward - spin_penalty - center_penalty

        # Override termination for infinite continuous learning
        return ALU_Q16.to_q(padded_obs), continuous_reward, False, False, info


def run_cartpole_boreal():
    print("🚀 Initializing BOREAL Engine for CartPole-v1...")

    raw_env = gym.make("CartPole-v1")
    # Disable Gymnasium internal termination bounds for infinite physics

    raw_env.unwrapped.theta_threshold_radians = 100 * math.pi
    raw_env.unwrapped.x_threshold = 10000.0

    env = CartPoleFixedPointWrapper(raw_env)

    ensemble = MetaEpistemicEnsemble()
    gate = TriStateGateQ16()

    states_q = [np.zeros(config.s_dim, dtype=np.int64) for _ in range(config.n_cores)]

    # Target observation: Cart at center, Pole upright, velocities zero
    target_obs = np.zeros(config.o_dim, dtype=np.float64)
    target_obs_q = ALU_Q16.to_q(target_obs)

    logger = ApexLogger()
    logger.log["x"] = []
    logger.log["y"] = []  # We'll use this for pole angle
    logger.log["surp"] = []
    logger.log["unc"] = []
    logger.log["regime"] = []
    logger.log["mode"] = []
    logger.log["ep_lens"] = []

    global_t = 0
    max_episodes = 5

    for ep in range(max_episodes):
        o_q, _ = env.reset()

        recovery_mode = False
        recovery_timer = 0

        print(f"\n--- Starting Episode {ep+1}/{max_episodes} ---")

        for t in range(500):
            if recovery_mode:
                active_lr = config.recov_lr_shift
                exp_shift = 2
                recovery_timer -= 1
            else:
                active_lr = config.base_lr_shift
                exp_shift = 1 if ensemble.regime_stability < 30 else -2

            a_q = q16_cem_plan(ensemble, states_q, target_obs_q, exp_shift)

            o_q_next, reward, terminated, truncated, _ = env.step(a_q)

            states_q, surp_sq_q, unc_sq_q = ensemble.step(
                states_q, a_q, o_q_next, active_lr
            )
            block, flags = gate.evaluate(surp_sq_q, a_q)

            if block and not recovery_mode and t > 50:
                print(f"  [T:{t}] 🛑 GATE TRIGGERED: Shock Isolated. Adapting...")
                recovery_mode = True
                recovery_timer = 20
                gate.reset()

            if recovery_mode and recovery_timer <= 0 and surp_sq_q < (ALU_Q16.ONE >> 1):
                recovery_mode = False

            # Log Pole Angle (index 2 of observation array)
            pole_angle = ALU_Q16.to_f(o_q_next[2])
            logger.log["x"].append(global_t)
            logger.log["y"].append(pole_angle)
            logger.log["surp"].append(np.sqrt(ALU_Q16.to_f(surp_sq_q)))
            logger.log["unc"].append(np.sqrt(ALU_Q16.to_f(unc_sq_q)))
            logger.log["regime"].append(ensemble.regime_hash)
            logger.log["mode"].append(1.0 if recovery_mode else 0.0)

            m_str = (
                "RECOVERING"
                if recovery_mode
                else ("FORAGING" if exp_shift > 0 else "TARGETING")
            )
            s_f = np.sqrt(ALU_Q16.to_f(surp_sq_q))

            # Print less frequently to avoid spam
            if t % 5 == 0:
                print(
                    f"[{m_str:<10}] EP:{ep+1} T:{t:03d} | Angle: {pole_angle:>5.2f} rad | Surp: {s_f:.3f} | Regime: [0x{ensemble.regime_hash}]"
                )

            o_q = o_q_next
            global_t += 1

            if terminated or truncated:
                print(f"💥 CartPole Terminated at Tick {t} (Global Tick {global_t})")
                logger.log["ep_lens"].append(t)
                states_q = [
                    np.zeros(config.s_dim, dtype=np.int64)
                    for _ in range(config.n_cores)
                ]
                break
        else:
            logger.log["ep_lens"].append(500)

    print("Done. Rendering...")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(4, 1, height_ratios=[2, 1, 0.8, 1])
        fig.suptitle(
            "Apex Sovereign V2: CartPole-v1 Benchmark", fontsize=16, fontweight="bold"
        )

        ax0 = fig.add_subplot(gs[0])
        ax0.plot(
            logger.log["x"],
            logger.log["y"],
            "c-",
            linewidth=2,
            label="Pole Angle (rad)",
        )
        ax0.axhline(0, color="lime", linestyle="--", label="Target (0 rad)")

        # Add a light area for the safe pole threshold (-0.209 to 0.209 rad is non-terminal)
        ax0.axhspan(-0.209, 0.209, color="lime", alpha=0.2)

        ax0.set_title("CartPole Balancing Trajectory (Across Episodes)")
        ax0.legend(loc="upper left")
        ax0.grid(True, alpha=0.3)

        ax1 = fig.add_subplot(gs[1])
        ax1.plot(
            logger.log["surp"], "m-", label="Pragmatic Surprise (||e||)", linewidth=1.5
        )
        ax1.plot(logger.log["unc"], "y--", label="Epistemic Uncertainty", linewidth=2)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            max_y = max(logger.log["surp"]) if len(logger.log["surp"]) > 0 else 1.0
            ax1.fill_between(
                range(len(logger.log["mode"])),
                0,
                max_y,
                where=np.array(logger.log["mode"]) > 0,
                color="red",
                alpha=0.2,
                label="Gate Recovery Mode",
            )
        ax1.set_title("Cognitive Integers & Thermodynamic Free Energy")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[2])
        unique_hashes = list(dict.fromkeys(logger.log["regime"]))
        hash_ids = [unique_hashes.index(h) for h in logger.log["regime"]]
        ax2.step(range(len(hash_ids)), hash_ids, "g-", linewidth=2)
        ax2.set_title("L3 Abstract Regime Memory (SHA-256 Hashes)")
        ax2.set_yticks(range(len(unique_hashes)))
        ax2.set_yticklabels([f"0x{h}" for h in unique_hashes])
        ax2.set_xlabel("Hardware Ticks")

        ax3 = fig.add_subplot(gs[3])
        ax3.plot(
            range(1, len(logger.log["ep_lens"]) + 1),
            logger.log["ep_lens"],
            "o-",
            color="orange",
            linewidth=2,
        )
        ax3.set_title("Episode Survival Length")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Ticks Survived")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("cartpole_boreal_results.png")
        print("Saved plot to cartpole_boreal_results.png")
    except ImportError:
        print("Matplotlib not installed. Skipping plot.")
        pass


if __name__ == "__main__":
    run_cartpole_boreal()
