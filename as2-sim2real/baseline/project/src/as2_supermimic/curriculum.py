"""Configure AS2 training stages and target difficulty."""

STAGE_TASKS = {
  "chase": "As2-Chase-Flat",
  "dodge_balls": "As2-Chase-Flat",
  "dodge_bars": "As2-Chase-Flat",
  "obstacles": "As2-Chase-Obstacles",
  "terrain": "As2-Chase-Terrain",
}


def configure_stage(cfg, stage: str, task: str) -> None:
  if STAGE_TASKS.get(stage) != task:
    raise ValueError(f"Stage {stage!r} requires task {STAGE_TASKS.get(stage)!r}, got {task!r}")
  command = cfg.commands["target"]
  target = command.target
  target.high_z = (0.38, 0.46)
  target.mid_z = (0.24, 0.33)
  target.low_z = (0.06, 0.13)
  target.high_bar_z = (0.39, 0.44)
  target.low_bar_z = (0.06, 0.09)
  target.leg_offset = (0.10, 0.19)
  target.throw_speed = (1.5, 2.5)
  target.throw_speed_final = (2.5, 4.0)
  target.bar_speed = (0.8, 1.2)
  target.bar_speed_final = (1.2, 2.2)
  target.throw_gap = (1.0, 2.0)
  target.min_time_to_robot = 0.8
  target.min_ball_flight = 0.6
  target.max_time_to_robot = 1.0
  target.spawn_dist = (1.0, 2.0)
  target.flee_min_speed = 0.1
  target.speed_max = (0.2, 0.4)
  target.speed_max_final = (1.0, 1.5)
  target.flee_ratio = (0.4, 0.6)
  target.flee_ratio_final = (0.6, 0.85)
  target.escape_speed = (1.5, 2.0)
  target.ou_sigma = 0.8
  target.kind_probs = (1.0, 0.0, 0.0)
  if stage == "chase":
    command.mode_probs = (1.0, 0.0, 0.0)
  elif stage == "dodge_balls":
    command.mode_probs = (0.3, 0.3, 0.4)
  elif stage == "dodge_bars":
    command.mode_probs = (0.3, 0.3, 0.4)
    target.kind_probs = (0.4, 0.3, 0.3)
  elif stage == "obstacles":
    command.mode_probs = (0.8, 0.0, 0.2)
    target.obstacles.slot_prob = 0.4
    target.obstacles.empty_prob = 0.25
    target.obstacles.gap = 0.8
    target.speed_max_final = (0.8, 1.2)
  elif stage == "terrain":
    # Train terrain locomotion before adding thrown bars.
    command.mode_probs = (1.0, 0.0, 0.0)
    target.kind_probs = (0.0, 0.5, 0.5)
    target.speed_max_final = (0.6, 1.0)
    target.bars_follow_ground = True
