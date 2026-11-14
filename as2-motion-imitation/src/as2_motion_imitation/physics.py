"""CPU MuJoCo backend for the AS2 model."""

from pathlib import Path

import mujoco
import numpy as np
from mjlab.entity import Entity

from as2_supermimic.robot.as2 import get_as2_robot_cfg
from as2_supermimic.robot.morphology import JOINT_NAMES, LIMB_LEGS, ROOT_BODY

DT = 0.02
STATE_FRAMES = ("forward-axis-v1", "lateral-axis-v2")


def make_model(contact_audit=False):
    spec = Entity(get_as2_robot_cfg()).spec
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1],
        contype=1, conaffinity=1, friction=[0.6, 0.005, 0.0001],
        rgba=[0.72, 0.75, 0.77, 1],
    )
    spec.worldbody.add_light(pos=[0, -2, 3], dir=[0, 0.4, -1])
    if contact_audit:
        for geom in spec.geoms:
            if geom.name.endswith("_collision"):
                spec.add_sensor(
                    name=f"audit_{geom.name}", type=mujoco.mjtSensor.mjSENS_CONTACT,
                    objtype=mujoco.mjtObj.mjOBJ_GEOM, objname=geom.name,
                    reftype=mujoco.mjtObj.mjOBJ_GEOM, refname="floor",
                    intprm=[9, 0, 1],
                )
    model = spec.compile()
    model.opt.timestep = 0.002
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = 10
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 540
    return model


class Physics:
    def __init__(self, model=None, state_frame="forward-axis-v1"):
        if state_frame not in STATE_FRAMES:
            raise ValueError("unknown state frame")
        self.state_frame = state_frame
        self.model = model if model is not None else make_model()
        self.data = mujoco.MjData(self.model)
        self.jids = np.array([self.model.joint(n).id for n in JOINT_NAMES])
        self.qadr = self.model.jnt_qposadr[self.jids]
        self.vadr = self.model.jnt_dofadr[self.jids]
        self.default_q = self.model.key(0).qpos[self.qadr].copy()
        self.act_joints = self.model.actuator_trnid[:, 0]
        self.act_order = np.array([list(self.jids).index(j) for j in self.act_joints])
        self.foot_ids = [self.model.site(n).id for n in LIMB_LEGS]
        self.body_names = tuple(self.model.body(i).name for i in range(1, self.model.nbody))
        self.base_id = self.model.body(ROOT_BODY).id
        self.reset()

    def reset(self, seed=0, perturb=0.0):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[self.qadr] += rng.uniform(-perturb, perturb, 12)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        # Let the PD controller settle the robot under gravity.
        for _ in range(150):
            self.step(np.zeros(12))
        return self.state()

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("action must contain 12 finite values")
        target = self.default_q + 0.25 * action
        self.data.ctrl[:] = target[self.act_order]
        for _ in range(round(DT / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def velocity(self, body_id):
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros_like(jp)
        mujoco.mj_jacBody(self.model, self.data, jp, jr, body_id)
        return jp @ self.data.qvel, jr @ self.data.qvel

    def state(self):
        d = self.data
        rotation = d.xmat[self.base_id].reshape(3, 3)
        if self.state_frame == "lateral-axis-v2":
            yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])
        else:
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        cy, sy = np.cos(yaw), np.sin(yaw)
        yaw_inverse = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
        base = d.xpos[self.base_id]
        lin, ang = self.velocity(self.base_id)
        hips = np.array([d.body(f"{leg}_thigh").xpos for leg in LIMB_LEGS])
        knees = np.array([d.body(f"{leg}_calf").xpos for leg in LIMB_LEGS])
        feet = d.site_xpos[self.foot_ids]
        state = np.concatenate([
            base[2:3], rotation.T @ [0, 0, -1], yaw_inverse @ lin,
            rotation.T @ ang, d.qpos[self.qadr] - self.default_q,
            d.qvel[self.vadr],
            *[((p - base) @ yaw_inverse.T).ravel() for p in (hips, knees, feet)],
        ])
        return state.astype(np.float32), np.array([*base[:2], yaw], np.float32)

    def snapshot(self):
        linear, angular = zip(*(self.velocity(i) for i in range(1, self.model.nbody)))
        return {
            "joint_pos": self.data.qpos[self.qadr].copy(),
            "joint_vel": self.data.qvel[self.vadr].copy(),
            "body_pos_w": self.data.xpos[1:].copy(),
            "body_quat_w": self.data.xquat[1:].copy(),
            "body_lin_vel_w": np.array(linear),
            "body_ang_vel_w": np.array(angular),
            "foot_pos_w": self.data.site_xpos[self.foot_ids].copy(),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
        }

    def diagnostics(self):
        feet = set(self.model.geom(f"{leg}_foot_collision").id for leg in LIMB_LEGS)
        floor = self.model.geom("floor").id
        touching = set()
        nonfoot = False
        for con in self.data.contact:
            if con.dist > 0:
                continue
            if floor in (con.geom1, con.geom2):
                other = con.geom2 if con.geom1 == floor else con.geom1
                touching.add(other)
                nonfoot |= other not in feet
        s, _ = self.state()
        q = self.data.qpos[self.qadr]
        limits = self.model.jnt_range[self.jids]
        return {
            "height": float(s[0]),
            "tilt": float(np.arccos(np.clip(-s[3], -1, 1))),
            "speed": float(np.linalg.norm(s[4:7])),
            "feet_contact": len(feet & touching),
            "nonfoot_contact": bool(nonfoot),
            "joint_limit": bool(((q < limits[:, 0]) | (q > limits[:, 1])).any()),
            "torque_saturation": float(np.mean(
                np.abs(self.data.actuator_force) >= self.model.actuator_forcerange[:, 1] * 0.99
            )),
        }


class Video:
    def __init__(self, physics, path, label):
        self.physics, self.path, self.label = physics, Path(path), label
        self.renderer = mujoco.Renderer(physics.model, height=540, width=960)
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.6, 135, -20
        self.frames = []

    def capture(self, time):
        from PIL import Image, ImageDraw

        self.camera.lookat[:] = self.physics.data.xpos[self.physics.base_id]
        self.renderer.update_scene(self.physics.data, camera=self.camera)
        frame = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, 960, 34), fill=(245, 245, 245))
        draw.text((12, 10), f"{self.label} | t={time:.2f}s", fill=(20, 20, 20))
        self.frames.append(np.array(frame))

    def close(self):
        import mediapy
        from PIL import Image

        self.path.parent.mkdir(parents=True, exist_ok=True)
        mediapy.write_video(str(self.path), self.frames, fps=25)
        decoded = mediapy.read_video(str(self.path))
        assert len(decoded) == len(self.frames) and np.ptp(decoded) > 50
        Image.fromarray(self.frames[len(self.frames) // 2]).save(self.path.with_suffix(".png"))
        self.renderer.close()
