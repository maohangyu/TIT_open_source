# run the baselines
#screen python main.py --algo ppo --device cuda:0 --log-folder ./log/
#screen python main.py --algo ofe_ppo --device cuda:1 --log-folder ./log/
#screen python main.py --algo d2rl_ppo --device cuda:2 --log-folder ./log/
#screen python main.py --algo resnet_ppo --device cuda:3 --log-folder ./log/





# hypertuning & run the online-ClassicControl
#nohup python hypertuning.py --env-name Acrobot-v1 --algo vanilla_tit_ppo --n-timesteps 100000 --device cuda:1 > ./log/TuningAcrobotVanilla.txt 2>&1 &
#nohup python hypertuning.py --env-name Acrobot-v1 --algo enhanced_tit_ppo --n-timesteps 100000 --device cuda:1 > ./log/TuningAcrobotEnhanced.txt 2>&1 &
#nohup python hypertuning.py --env-name CartPole-v1 --algo vanilla_tit_ppo --n-timesteps 100000 --device cuda:2 > ./log/TuningCartPoleVanilla.txt 2>&1 &
#nohup python hypertuning.py --env-name CartPole-v1 --algo enhanced_tit_ppo --n-timesteps 100000 --device cuda:2 > ./log/TuningCartPoleEnhanced.txt 2>&1 &
#nohup python hypertuning.py --env-name MountainCar-v0 --algo vanilla_tit_ppo --n-timesteps 100000 --device cuda:3 > ./log/TuningMountainCarVanilla.txt 2>&1 &
#nohup python hypertuning.py --env-name MountainCar-v0 --algo enhanced_tit_ppo --n-timesteps 100000 --device cuda:3 > ./log/TuningMountainCarEnhanced.txt 2>&1 &

#nohup python main.py --env-name Acrobot-v1 --algo vanilla_tit_ppo --device cuda:1 > ./log/RunningAcrobotVanilla.txt 2>&1 &
#nohup python main.py --env-name Acrobot-v1 --algo enhanced_tit_ppo --device cuda:1 > ./log/RunningAcrobotEnhanced.txt 2>&1 &
#nohup python main.py --env-name CartPole-v1 --algo vanilla_tit_ppo --device cuda:2 > ./log/RunningCartPoleVanilla.txt 2>&1 &
#nohup python main.py --env-name CartPole-v1 --algo enhanced_tit_ppo --device cuda:2 > ./log/RunningCartPoleEnhanced.txt 2>&1 &
#nohup python main.py --env-name MountainCar-v0 --algo vanilla_tit_ppo --device cuda:3 > ./log/RunningMountainCarVanilla.txt 2>&1 &
#nohup python main.py --env-name MountainCar-v0 --algo enhanced_tit_ppo --device cuda:3 > ./log/RunningMountainCarEnhanced.txt 2>&1 &





# hypertuning & run the online-MuJoCo
#nohup python -u hypertuning.py --env-name Ant-v3 --algo vanilla_tit_ppo --n-timesteps 1000000 --device cuda:1 > ./log/TuningAntVanilla.txt 2>&1 &
#nohup python -u hypertuning.py --env-name Ant-v3 --algo enhanced_tit_ppo --n-timesteps 1000000 --device cuda:1 > ./log/TuningAntEnhanced.txt 2>&1 &
#nohup python -u hypertuning.py --env-name Hopper-v3 --algo vanilla_tit_ppo --n-timesteps 1000000 --device cuda:2 > ./log/TuningHopperVanilla.txt 2>&1 &
#nohup python -u hypertuning.py --env-name Hopper-v3 --algo enhanced_tit_ppo --n-timesteps 1000000 --device cuda:2 > ./log/TuningHopperEnhanced.txt 2>&1 &
#nohup python -u hypertuning.py --env-name Walker2d-v3 --algo vanilla_tit_ppo --n-timesteps 1000000 --device cuda:3 > ./log/TuningWalker2dVanilla.txt 2>&1 &
#nohup python -u hypertuning.py --env-name Walker2d-v3 --algo enhanced_tit_ppo --n-timesteps 1000000 --device cuda:3 > ./log/TuningWalker2dEnhanced.txt 2>&1 &

#nohup python main.py --env-name Ant-v3 --algo vanilla_tit_ppo --device cuda:1 > ./log/RunningAntVanilla.txt 2>&1 &
#nohup python main.py --env-name Ant-v3 --algo enhanced_tit_ppo --device cuda:1 > ./log/RunningAntEnhanced.txt 2>&1 &
#nohup python main.py --env-name Hopper-v3 --algo vanilla_tit_ppo --device cuda:2 > ./log/RunningHopperVanilla.txt 2>&1 &
#nohup python main.py --env-name Hopper-v3 --algo enhanced_tit_ppo --device cuda:2 > ./log/RunningHopperEnhanced.txt 2>&1 &
#nohup python main.py --env-name Walker2d-v3 --algo vanilla_tit_ppo --device cuda:3 > ./log/RunningWalker2dVanilla.txt 2>&1 &
#nohup python main.py --env-name Walker2d-v3 --algo enhanced_tit_ppo --device cuda:3 > ./log/RunningWalker2dEnhanced.txt 2>&1 &





# hypertuning & run the online-Atari
#nohup python hypertuning.py --env-name BreakoutNoFrameskip-v4 --algo vanilla_tit_ppo --n-timesteps 1000000 --device cuda:0 > ./log/TuningBreakoutVanilla.txt 2>&1 &
#nohup python hypertuning.py --env-name BreakoutNoFrameskip-v4 --algo enhanced_tit_ppo --n-timesteps 1000000 --device cuda:0 > ./log/TuningBreakoutEnhanced.txt 2>&1 &
#nohup python hypertuning.py --env-name MsPacmanNoFrameskip-v4 --algo vanilla_tit_ppo --n-timesteps 1000000 --device cuda:1 > ./log/TuningMsPacmanVanilla.txt 2>&1 &
#nohup python hypertuning.py --env-name MsPacmanNoFrameskip-v4 --algo enhanced_tit_ppo --n-timesteps 1000000 --device cuda:1 > ./log/TuningMsPacmanEnhanced.txt 2>&1 &
#nohup python hypertuning.py --env-name PongNoFrameskip-v4 --algo vanilla_tit_ppo --n-timesteps 1000000 --device cuda:2 > ./log/TuningPongVanilla.txt 2>&1 &
#nohup python hypertuning.py --env-name PongNoFrameskip-v4 --algo enhanced_tit_ppo --n-timesteps 1000000 --device cuda:2 > ./log/TuningPongEnhanced.txt 2>&1 &
#nohup python hypertuning.py --env-name SpaceInvadersNoFrameskip-v4 --algo vanilla_tit_ppo --n-timesteps 1000000 --device cuda:3 > ./log/TuningSpaceInvadersVanilla.txt 2>&1 &
#nohup python hypertuning.py --env-name SpaceInvadersNoFrameskip-v4 --algo enhanced_tit_ppo --n-timesteps 1000000 --device cuda:3 > ./log/TuningSpaceInvadersEnhanced.txt 2>&1 &

#nohup python main.py --env-name BreakoutNoFrameskip-v4 --algo vanilla_tit_ppo --device cuda:0 > ./log/RunningBreakoutVanilla.txt 2>&1 &
#nohup python main.py --env-name BreakoutNoFrameskip-v4 --algo enhanced_tit_ppo --device cuda:0 > ./log/RunningBreakoutEnhanced.txt 2>&1 &
#nohup python main.py --env-name MsPacmanNoFrameskip-v4 --algo vanilla_tit_ppo --device cuda:1 > ./log/RunningMsPacmanVanilla.txt 2>&1 &
#nohup python main.py --env-name MsPacmanNoFrameskip-v4 --algo enhanced_tit_ppo --device cuda:1 > ./log/RunningMsPacmanEnhanced.txt 2>&1 &
#nohup python main.py --env-name PongNoFrameskip-v4 --algo vanilla_tit_ppo --device cuda:2 > ./log/RunningPongVanilla.txt 2>&1 &
#nohup python main.py --env-name PongNoFrameskip-v4 --algo enhanced_tit_ppo --device cuda:2 > ./log/RunningPongEnhanced.txt 2>&1 &
#nohup python main.py --env-name SpaceInvadersNoFrameskip-v4 --algo vanilla_tit_ppo --device cuda:3 > ./log/RunningSpaceInvadersVanilla.txt 2>&1 &
#nohup python main.py --env-name SpaceInvadersNoFrameskip-v4 --algo enhanced_tit_ppo --device cuda:3 > ./log/RunningSpaceInvadersEnhanced.txt 2>&1 &





# hypertuning & run the offline-MuJoCo-medium
#nohup python -u offline_hypertuning.py --env-name halfcheetah-medium-v0 --algo enhanced_tit_cql --n-timesteps 500000 --device 1 > ./log/TuningHalfcheetahMediumEnhanced.txt 2>&1 &
#nohup python -u offline_hypertuning.py --env-name hopper-medium-v0 --algo enhanced_tit_cql --n-timesteps 500000 --device 2 > ./log/TuningHopperMediumEnhanced.txt 2>&1 &
#nohup python -u offline_hypertuning.py --env-name walker2d-medium-v0 --algo enhanced_tit_cql --n-timesteps 500000 --device 3 > ./log/TuningWalker2dMediumEnhanced.txt 2>&1 &

#nohup python offline_main.py --env-name halfcheetah-medium-v0 --algo enhanced_tit_cql --device 1 > ./log/RunningHalfcheetahMediumEnhanced.txt 2>&1 &
#nohup python offline_main.py --env-name hopper-medium-v0 --algo enhanced_tit_cql --device 2 > ./log/RunningHopperMediumEnhanced.txt 2>&1 &
#nohup python offline_main.py --env-name walker2d-medium-v0 --algo enhanced_tit_cql --device 3 > ./log/RunningWalker2dMediumEnhanced.txt 2>&1 &





# hypertuning & run the offline-MuJoCo-medium-replay
#nohup python -u offline_hypertuning.py --env-name halfcheetah-medium-replay-v0 --algo enhanced_tit_cql --n-timesteps 500000 --device 1 > ./log/TuningHalfcheetahMediumReplayEnhanced.txt 2>&1 &
#nohup python -u offline_hypertuning.py --env-name hopper-medium-replay-v0 --algo enhanced_tit_cql --n-timesteps 500000 --device 2 > ./log/TuningHopperMediumReplayEnhanced.txt 2>&1 &
#nohup python -u offline_hypertuning.py --env-name walker2d-medium-replay-v0 --algo enhanced_tit_cql --n-timesteps 500000 --device 3 > ./log/TuningWalker2dMediumReplayEnhanced.txt 2>&1 &

#nohup python offline_main.py --env-name halfcheetah-medium-replay-v0 --algo enhanced_tit_cql --device 1 > ./log/RunningHalfcheetahMediumReplayEnhanced.txt 2>&1 &
#nohup python offline_main.py --env-name hopper-medium-replay-v0 --algo enhanced_tit_cql --device 2 > ./log/RunningHopperMediumReplayEnhanced.txt 2>&1 &
#nohup python offline_main.py --env-name walker2d-medium-replay-v0 --algo enhanced_tit_cql --device 3 > ./log/RunningWalker2dMediumReplayEnhanced.txt 2>&1 &

