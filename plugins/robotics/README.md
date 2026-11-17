# Robotics

## Advancing accessible robotics

The Robotics plugin brings frontier spatial reasoning, multimodal perception,
and code generation directly into the real-time robot loop. Your agent writes
the controller, tests it in [MuJoCo](https://mujoco.org) simulation, and runs
the same code on a real robot arm. You can also work in simulation only, with
no robot hardware. This makes closed-loop control, rapid hardware bring-up, and
continuous robotics applications fast and affordable enough for everyday use.

<p align="center">
  <img src="assets/infographic.png" alt="Robotics plugin loop: bring up, perceive, plan, act, and verify, with the same controller in MuJoCo simulation and on the real robot" width="100%" />
</p>

## Examples

### Cookie factory

An I2RT YAM Pro arm sorts cookies from a moving conveyor belt into two boxes.

Simulation (MuJoCo) | Real robot
:-----------------: | :--------:
<img src="assets/cookie_factory_sim.gif" alt="Cookie sorting in MuJoCo simulation" width="420" /> | <img src="assets/cookie_factory_real.gif" alt="Cookie sorting on a real I2RT YAM Pro arm" width="420" />

## Example prompts

### Simulation only (no robot needed)

*   *"Generate a pelican riding a bicycle simulation in MuJoCo."*
*   *"Create a MuJoCo grocery sorting task using a LeRobot SO-101 arm.
    Scatter groceries of different shapes and sizes across a countertop, and
    have the robot sort them into designated bins without crushing fragile items
    or knocking anything over. Include randomized layouts. Show me some example
    successful trajectories."*
*   *"Build a realistic coffee shop scene in MuJoCo with bimanual Franka
    Research 3 robot arms, espresso machine, cups, and serving counter. Program
    the robot to pick up an empty cup, place it beneath the coffee dispenser,
    and deliver the finished drink onto a tray. All object movements must be
    driven by contact from the robot."*
*   *"In simulation, build my setup from these photos: one YAM Pro arm, a
    moving conveyor belt, two types of cookies, and two boxes. Use the YAM Pro
    arm to pick each cookie off the moving belt and place it in the matching
    box. Test it across several runs and show me a video."*

### Real robot

*   *"Bring up my real SO-101 arm and use it to pick up the blue marker
    from the table. Record a video."*
*   *"Bring up my real YAM arm and use it to wave hi. Record a video."*
*   *"Bring up my real YAM arm and use it to place the pen several inches
    deep inside the tall box without touching the walls. Record a video."*

## Safety on real robots

> [!WARNING]
> A moving robot can injure people and damage equipment. Stay next to the
> robot, keep the workspace clear, and keep a way to cut power within reach.
> You are responsible for the safe operation of your robot.

## Learn more

You can find more about Gemini Robotics, including demos, at
[deepmind.google/models/gemini-robotics](https://deepmind.google/models/gemini-robotics/).
Share your examples and connect with other builders in
[GitHub Discussions](https://github.com/google-deepmind/robotics-antigravity-plugin/discussions).

## License

Copyright 2026 Google LLC. Licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Some skills
download third-party software and robot models, or call third-party services
such as the Gemini API, at run time. These are licensed by their owners under
their own terms.
