#nohup python -u experiment.py --env hopper --dataset medium --model_type tit --device cuda:1 > ./log/hopper-medium-tit_log.txt 2>&1 &
#nohup python -u experiment.py --env hopper --dataset medium --model_type dt --device cuda:2 > ./log/hopper-medium-dt_log.txt 2>&1 &
#nohup python -u experiment.py --env walker2d --dataset medium --model_type tit --device cuda:1 > ./log/walker2d-medium-tit_log.txt 2>&1 &
#nohup python -u experiment.py --env walker2d --dataset medium --model_type dt --device cuda:2 > ./log/walker2d-medium-dt_log.txt 2>&1 &
#nohup python -u experiment.py --env halfcheetah --dataset medium --model_type tit --device cuda:1 > ./log/halfcheetah-medium-tit_log.txt 2>&1 &
#nohup python -u experiment.py --env halfcheetah --dataset medium --model_type dt --device cuda:2 > ./log/halfcheetah-medium-dt_log.txt 2>&1 &





#nohup python -u experiment.py --env hopper --dataset medium-replay --model_type tit --device cuda:1 > ./log/hopper-medium-replay-tit_log.txt 2>&1 &
#nohup python -u experiment.py --env hopper --dataset medium-replay --model_type dt --device cuda:2 > ./log/hopper-medium-replay-dt_log.txt 2>&1 &
#nohup python -u experiment.py --env walker2d --dataset medium-replay --model_type tit --device cuda:3 > ./log/walker2d-medium-replay-tit_log.txt 2>&1 &
#nohup python -u experiment.py --env walker2d --dataset medium-replay --model_type dt --device cuda:3 > ./log/walker2d-medium-replay-dt_log.txt 2>&1 &
#nohup python -u experiment.py --env halfcheetah --dataset medium-replay --model_type tit --device cuda:3 > ./log/halfcheetah-medium-replay-tit_log.txt 2>&1 &
#nohup python -u experiment.py --env halfcheetah --dataset medium-replay --model_type dt --device cuda:3 > ./log/halfcheetah-medium-replay-dt_log.txt 2>&1 &
