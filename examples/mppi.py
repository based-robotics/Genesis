import torch
import os

# os.environ["PYOPENGL_PLATFORM"] = "glx"  # or try 'glx' if this doesn't work
import genesis as gs
import genesis.engine.entities as gsen

gs.init(backend=gs.gpu)
viewer_options = gs.options.ViewerOptions(
    camera_pos=(2.5, 0.0, 1.5),
    camera_lookat=(0.0, 0.0, 0.5),
    camera_fov=30,
    max_FPS=60,
)


class MPPI:
    N_SAMPLES = 100
    HORIZON = 30
    # N_SAMPLES = 100
    # HORIZON = 50
    NOISE_SIGMA = 20.0
    ACT_DIM = 9
    STATE_DIM = 9 + 9
    TEMPERATURE = 1.0

    def __init__(self):
        self.plan = torch.zeros(self.HORIZON, self.ACT_DIM, device=gs.device)

        scene = gs.Scene(
            show_viewer=False,
            rigid_options=gs.options.RigidOptions(
                dt=0.01,
            ),
            viewer_options=viewer_options,
            show_FPS=False,
        )

        scene.add_entity(
            gs.morphs.Plane(),
        )
        self.robot: gsen.RigidEntity = scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        )
        scene.build(n_envs=self.N_SAMPLES, env_spacing=(1.0, 1.0))
        self.scene = scene

    def _get_obs(self):
        pos = self.robot.get_dofs_position()
        vel = self.robot.get_dofs_velocity()

        return torch.cat([pos, vel], dim=-1)

    def _sample_noise(self):
        size = (self.HORIZON, self.N_SAMPLES, self.ACT_DIM)
        return torch.randn(size, device=gs.device) * self.NOISE_SIGMA

    def rollout_fn(self, obs, acts):
        # we need to simulate the environment
        self.robot.set_dofs_position(obs[: self.ACT_DIM])
        self.robot.set_dofs_velocity(obs[self.ACT_DIM :])

        rollout_recording = torch.zeros(
            self.HORIZON,
            self.N_SAMPLES,
            self.STATE_DIM,
            device=gs.device,
        )
        for t in range(self.HORIZON):
            self.robot.control_dofs_force(acts[t])
            self.scene.step()
            rollout_recording[t] = self._get_obs()

        return rollout_recording

    def _compute_cost(self, state_sequences, action_sequences):
        # we want to arrive at a target joint configuration
        q_target = (
            torch.tensor(
                [1, -1, 0.4, -0.4, 0.4, 0.4, 0.4, 0.02, 0.02],
                device=gs.device,
            )
            * 0.0
        )

        q = state_sequences[
            :, :, : self.ACT_DIM
        ]  # Shape: (HORIZON, N_SAMPLES, ACT_DIM)

        # Calculate the L2 distance to the target configuration
        cost_to_target = torch.sum(
            (q - q_target) ** 2, dim=-1
        )  # Shape: (HORIZON, N_SAMPLES)

        # Optional: include action cost (e.g., regularization to minimize actions)
        action_cost = torch.sum(
            action_sequences**2, dim=-1
        )  # Shape: (HORIZON, N_SAMPLES)

        # Combine costs (weighted sum or other aggregation)
        total_cost = torch.sum(cost_to_target, dim=0) + 0 * torch.sum(
            action_cost, dim=0
        )  # Shape: (N_SAMPLES)

        return total_cost

    def get_action(self, obs):
        env_plan = torch.tile(self.plan, (self.N_SAMPLES, 1, 1)).permute(
            1, 0, 2
        )  # Shape: (HORIZON, N_SAMPLES, ACT_DIM)
        noise = self._sample_noise()

        # print(env_plan.shape, noise.shape)

        acts = env_plan + noise
        acts = acts.clamp(-100.0, 100.0)  # TODO: update LB and UB

        rollout_recording = self.rollout_fn(obs, acts)
        costs = self._compute_cost(rollout_recording, acts)

        exp_costs = torch.exp(self.TEMPERATURE * (costs.min() - costs))
        denom = exp_costs.sum() + 1e-10

        weighted_inputs = exp_costs[None, :, None] * acts
        new_plan = weighted_inputs.sum(dim=1) / denom

        self.plan = new_plan

        # return only the first action
        return new_plan[0]


def create_sim():
    scene = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
        ),
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )

    robot: gsen.RigidEntity = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    scene.build()

    return scene, robot


def get_obs(robot):
    pos = robot.get_dofs_position()
    vel = robot.get_dofs_velocity()

    return torch.cat([pos, vel], dim=-1)


# scene, robot = create_sim()

# pos = robot.get_dofs_position()
# vel = robot.get_dofs_velocity()

# print(pos.shape, vel.shape)


import zenoh
import json

session = zenoh.open(zenoh.Config())

current_obs = torch.zeros(18, device=gs.device)


def listener(sample):
    strdata = sample.payload.to_bytes().decode("utf-8")
    js = json.loads(strdata)
    last_state = js["obs"]

    current_obs.copy_(torch.tensor(last_state, device=gs.device))

    # print(last_state)


sub = session.declare_subscriber("panda_obs", listener)
pub = session.declare_publisher("panda_ctrl")
import time

# time.sleep(100)

mppi = MPPI()

# for _ in range(1000):
while True:
    # obs = get_obs(robot)
    # obs = torch.zeros(18, device=gs.device)
    current_action = mppi.get_action(current_obs)
    print(f"action: {current_action}")

    pub.put(
        json.dumps({"ctrl": current_action.cpu().numpy().tolist()}),
        encoding=zenoh.Encoding.APPLICATION_JSON,
    )

    # robot.control_dofs_force(current_action)
    # scene.step()
    # print("current qpos is ", robot.get_dofs_position())
    # break
