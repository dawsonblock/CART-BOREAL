import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import threading

import sys
import os
import math
import numpy as np
import gymnasium as gym

# Import the existing sovereign engine
from boreal_cartpole import CartPoleFixedPointWrapper
from boreal_apex_sovereign_v2 import (
    ALU_Q16,
    MetaEpistemicEnsemble,
    TriStateGateQ16,
    q16_cem_plan,
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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


class LiveCartPoleEngine:
    def __init__(self, broadcast_callback):
        self.broadcast = broadcast_callback
        self.is_running = False

        self.raw_env = gym.make("CartPole-v1")
        # Disable Gymnasium internal termination bounds for infinite physics
        self.raw_env.unwrapped.theta_threshold_radians = 100 * math.pi
        self.raw_env.unwrapped.x_threshold = 10000.0

        self.env = CartPoleFixedPointWrapper(self.raw_env)

        self.ensemble = MetaEpistemicEnsemble()
        self.gate = TriStateGateQ16()

        self.states_q = [
            np.zeros(config.s_dim, dtype=np.int64) for _ in range(config.n_cores)
        ]

        # Target observation: Cart at center, Pole upright, velocities zero
        target_obs = np.zeros(config.o_dim, dtype=np.float64)
        self.target_obs_q = ALU_Q16.to_q(target_obs)

    async def live_run(self):
        import logging

        logging.basicConfig(
            level=logging.INFO,
            filename="cartpole_run.log",
            filemode="w",
            format="%(asctime)s - %(message)s",
        )
        logging.info("Starting CartPole live run...")

        self.is_running = True

        o_q, _ = self.env.reset()

        recovery_mode = False
        recovery_timer = 0
        t = 0
        global_t = 0
        ep = 1

        while self.is_running:
            if recovery_mode:
                active_lr = config.recov_lr_shift
                exp_shift = 2
                recovery_timer -= 1
            else:
                active_lr = config.base_lr_shift
                exp_shift = 1 if self.ensemble.regime_stability < 30 else -2

            a_q = q16_cem_plan(
                self.ensemble, self.states_q, self.target_obs_q, exp_shift
            )

            o_q_next, reward, terminated, truncated, _ = self.env.step(a_q)

            self.states_q, surp_sq_q, unc_sq_q = self.ensemble.step(
                self.states_q, a_q, o_q_next, active_lr
            )
            block, flags = self.gate.evaluate(surp_sq_q, a_q)

            if block and not recovery_mode and t > 50:
                recovery_mode = True
                recovery_timer = 20
                self.gate.reset()

            if recovery_mode and recovery_timer <= 0 and surp_sq_q < (ALU_Q16.ONE * 5):
                recovery_mode = False

            # Broadcast State
            x_val, _, theta_val, _ = self.env.unwrapped.state
            pole_angle = float(theta_val)
            cart_pos = float(x_val)

            state_data = {
                "tick": global_t,
                "ep": ep,
                "ep_tick": t,
                "x": float(cart_pos),
                "y": float(
                    pole_angle
                ),  # We'll map pole angle to Y for the generic UI map
                "surprise": float(np.sqrt(ALU_Q16.to_f(surp_sq_q))),
                "uncertainty": float(np.sqrt(ALU_Q16.to_f(unc_sq_q))),
                "regime": self.ensemble.regime_hash,
                "mode": (
                    "RECOVERING"
                    if recovery_mode
                    else ("FORAGING" if exp_shift > 0 else "TARGETING")
                ),
                "gate_blocked": bool(block),
                "terminated": bool(terminated or truncated),
            }

            if t % 50 == 0:
                logging.info(
                    f"Tick: {t} | x: {cart_pos:.3f}, theta: {pole_angle:.3f}, surp: {state_data['surprise']:.3f}, unc: {state_data['uncertainty']:.3f}, mode: {state_data['mode']}"
                )

            await self.broadcast(json.dumps(state_data))

            o_q = o_q_next
            t += 1
            global_t += 1

            await asyncio.sleep(0.02)  # 50 FPS


global_engine_task = None


async def broadcast_callback(msg):
    await manager.broadcast(msg)


@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data)

            global global_engine_task

            if cmd.get("action") == "start":
                if global_engine_task is None or global_engine_task.done():
                    engine = LiveCartPoleEngine(broadcast_callback)
                    global_engine_task = asyncio.create_task(engine.live_run())
            elif cmd.get("action") == "stop":
                if global_engine_task and not global_engine_task.done():
                    global_engine_task.cancel()

    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
