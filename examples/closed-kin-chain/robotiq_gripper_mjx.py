import mujoco as mj
from mujoco import viewer


def main():
    gripper_path = "mujoco_menagerie/robotiq_2f85_v4/scene.xml"
    mj_model = mj.MjModel.from_xml_path(gripper_path)

    mj_data = mj.MjData(mj_model)
    mj_viewer = viewer.launch_passive(
        mj_model,
        mj_data,
        show_left_ui=False,
        show_right_ui=False,
    )

    for i in range(2):
        mj.mj_step(mj_model, mj_data)
        print(mj_data.efc_aref)
        print(mj_data.)
        mj_viewer.sync()


if __name__ == "__main__":
    main()
