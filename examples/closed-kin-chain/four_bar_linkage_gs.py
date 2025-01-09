import argparse
import os
import pickle as pkl

import numpy as np

import genesis as gs
import genesis.engine.entities as gsen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu, debug=args.debug)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(1.0, -1.5, 1.5),
        camera_lookat=(0.3, 0.0, 1.0),
        camera_fov=45,
        max_FPS=60,
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.001,
        ),
        viewer_options=viewer_options,
        show_viewer=args.vis,
        show_FPS=False,
    )

    # print(scene.rigid_solver._options)

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    four_bar: gsen.RigidEntity = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="four_bar_linkage.xml",
        ),
    )

    ########################## build ##########################
    scene.build()
    for i in range(len(scene.rigid_solver.links)):
        print(f"Entity with id {scene.rigid_solver.links[i].idx}: {scene.rigid_solver.links[i].name}")

    # Get joint IDs for the actuated joints
    j1_id = four_bar.get_joint("joint1").dof_start
    j3_id = four_bar.get_joint("joint3").dof_start

    # Simulation loop with sinusoidal torques
    for i in range(1000000000):
        # Apply sinusoidal torques with different phases
        # torque1 = 0*np.sin(i / 20) * 30
        # torque2 = 0*np.sin(i / 20 + np.pi/2) * 30  # Phase shifted
        
        # four_bar.control_dofs_force(np.array([torque1, torque2]), [j1_id, j3_id])
        scene.step()


if __name__ == "__main__":
    main() 