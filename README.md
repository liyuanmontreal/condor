
rl
1.generate offline data：
python experiments/build_rl_dataset_2d.py

2.train 2D FQI policy：
python experiments/run_fqi_2d_training.py
generate：outputs/fqi_2d_model.pkl

3.generate 30 年simulation：
python experiments/run_fqi_2d_policy_simulation.py


mpc
python experiments/run_mpc_dynamic_target.py
