
# 1. Quick sanity check, a few minutes
python -m panel.experiments.h1_compare --seeds 1 --ppo-steps 60000 --panel-steps 8000 --out runs/smoke

# 2. Real H1 run on DoorKey-5x5, 3 seeds
python -m panel.experiments.h1_compare --env MiniGrid-DoorKey-5x5-v0 --seeds 3 --ppo-steps 300000 --panel-steps 100000 --out runs/h1_doorkey5

# 3. The plan's target env (DoorKey-8x8) once step 2 looks sane
python -m panel.experiments.h1_compare --env MiniGrid-DoorKey-8x8-v0 --seeds 3 --ppo-steps 1000000 --panel-steps 300000 --out runs/h1_doorkey8

# Optional: redo the plot from saved logs, and run the slow PPO learning test
python -m panel.analysis.plot runs/h1_doorkey5
python -m pytest -m slow
