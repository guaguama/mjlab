"""Stage 1.5 of the OSL plan: the human actor beside the OSL's walking controller.

Stage 2 (:mod:`mjlab.tasks.osl_split`) splits stage 1's unified policy into a human
actor and a learned OSL actor that sees only the prosthesis's own sensors. Stage 1.5
is stage 2 with no OSL network: the controller the device runs drives the OSL.

* the **OSL** runs the classical four-state impedance controller of the
  opensourceleg SDK and MyoSuite (early and late stance, early and late swing), as
  osl_deploy's ``osl_fsm`` runs it: once per 100 Hz control step it reads the clean
  knee and ankle encoders and the tared axial ankle load, takes at most one
  transition and commands the new state's spring-damper torque, held over the
  physics substeps, through the nominal motor gains and with no motor delay. Its
  plant matches osl_deploy's step for step, so a human trained here meets in
  deployment the controller it trained beside;
* the **human actor** is stage 2's: the stage-1 actor minus its two OSL rows (an
  exact transplant), with the stage-1 critic, learning from the first iteration on
  stage 2's rate, held std and reward channel. Only the human learns;
* physics randomization (masses, friction, motor strength, pushes, the human's
  delay) is stage 1's, as in stage 2; the controller's inputs are exact.

``fsm`` holds the batched controller, ``mdp`` its action term, ``config`` the env
config, the runner settings and the task registration
(``Mjlab-FSM-Unicycle-OSL-Fullbody-{ROBOT,MSK}[-CLF][-Compliant]``), and ``rl`` the
human-only PPO, its inference policy and the runner.
"""
