"""Stage 2 of the OSL plan: separate human and OSL actors over the gait library.

Stage 1 (:mod:`mjlab.tasks.osl_gait_library`) trains one unified policy that drives
the whole model, human actuators and OSL motors alike, from privileged observations.
Stage 2 splits it:

* the **human actor** is the unified actor minus its two OSL output rows (an exact
  transplant), with the unified critic as its critic. It starts frozen for a number of
  iterations and keeps the stage-1 action std;
* the **OSL actor** is a fresh, small network that sees only the prosthesis's own
  sensors (knee/ankle encoders, a shank IMU, load cells and its own previous command),
  with an observation history. It never sees the reference, the commanded twist or
  the stride clock, and learns to follow them only through its reward;
* each agent has its own critic (both full-state), PPO ratio and clip, optimizer and
  reward channel. The channels re-weight the stage-1 imitation reward's coordinates,
  so at the stage-1 weights both equal the stage-1 reward;
* physics randomization is stage 1's; perception randomization (sensor noise, bias,
  latency) applies to the OSL actor's observations only.

``config`` registers the tasks (``Mjlab-Split-Unicycle-OSL-Fullbody[-MSK][-CLF]
[-Compliant]``), ``mdp`` holds the OSL sensors' observation terms and the two-channel
reward, ``rl`` the two-agent PPO, its runner and the stage-1 warm-start converter, and
``scripts`` the converter's command line and the two-stage pipeline driver.
"""
