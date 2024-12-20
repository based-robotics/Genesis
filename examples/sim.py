import torch
import os
import zenoh
import json

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


def create_sim():
    scene = gs.Scene(
        show_viewer=True,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
        ),
        viewer_options=viewer_options,
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


scene, robot = create_sim()

pos = robot.get_dofs_position()
vel = robot.get_dofs_velocity()

last_ctrl = torch.zeros(9, device=gs.device)


def listener(sample):
    # decode to json
    # ctrl = json.loads(sample.payload.decode("utf-8"))["ctrl"]
    strdata = sample.payload.to_bytes().decode("utf-8")
    data = json.loads(strdata)
    ctrl = data["ctrl"]

    ctrl = torch.tensor(ctrl, device=gs.device)

    last_ctrl.copy_(ctrl)


session = zenoh.open(zenoh.Config())
sub = session.declare_subscriber("panda_ctrl", listener)
pub = session.declare_publisher("panda_obs")

# wait for the first control
while last_ctrl.sum() == 0:
    pass

while scene.t < 100000.0:
    # set last control
    print(f"new control: {last_ctrl}")
    # robot.set_dofs_position(pos)
    robot.control_dofs_force(last_ctrl)

    scene.step()

    answer = {"obs": get_obs(robot).cpu().numpy().tolist()}
    pub.put(json.dumps(answer), encoding=zenoh.Encoding.APPLICATION_JSON)
