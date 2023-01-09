#nohup python run_dt_atari.py --seed 123 --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128 > ./log/Breakout-dt_log_123.txt 2>&1 &
#nohup python run_dt_atari.py --seed 231 --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128 > ./log/Breakout-dt_log_231.txt 2>&1 &
#nohup python run_dt_atari.py --seed 312 --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128 > ./log/Breakout-dt_log_312.txt 2>&1 &

#nohup python run_dt_atari.py --seed 123 --context_length 50 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512 > ./log/Pong-dt_log_123.txt 2>&1 &
#nohup python run_dt_atari.py --seed 231 --context_length 50 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512 > ./log/Pong-dt_log_231.txt 2>&1 &
#nohup python run_dt_atari.py --seed 312 --context_length 50 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512 > ./log/Pong-dt_log_312.txt 2>&1 &



#nohup python run_dt_atari.py --seed 123 --context_length 30 --epochs 5 --model_type 'reward_conditioned_tit' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128 > ./log/Breakout-tit_log_123.txt 2>&1 &
#nohup python run_dt_atari.py --seed 231 --context_length 30 --epochs 5 --model_type 'reward_conditioned_tit' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128 > ./log/Breakout-tit_log_231.txt 2>&1 &
#nohup python run_dt_atari.py --seed 312 --context_length 30 --epochs 5 --model_type 'reward_conditioned_tit' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128 > ./log/Breakout-tit_log_312.txt 2>&1 &

#nohup python run_dt_atari.py --seed 123 --context_length 50 --epochs 5 --model_type 'reward_conditioned_tit' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512 > ./log/Pong-tit_log_123.txt 2>&1 &
#nohup python run_dt_atari.py --seed 231 --context_length 50 --epochs 5 --model_type 'reward_conditioned_tit' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512 > ./log/Pong-tit_log_231.txt 2>&1 &
#nohup python run_dt_atari.py --seed 312 --context_length 50 --epochs 5 --model_type 'reward_conditioned_tit' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512 > ./log/Pong-tit_log_312.txt 2>&1 &




# Decision Transformer (DT)
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Qbert' --batch_size 128
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 50 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Seaquest' --batch_size 128
#done
#
## Behavior Cloning (BC)
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Qbert' --batch_size 128
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 50 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512
#done
#
#for seed in 123 231 312
#do
#    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Seaquest' --batch_size 128
#done