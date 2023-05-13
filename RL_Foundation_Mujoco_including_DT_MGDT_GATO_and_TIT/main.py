"""
highly based on https://github.com/kzl/decision-transformer/blob/master/gym/experiment.py#L208
"""

# import wandb
from tensorboardX import SummaryWriter
import torch

import argparse
import yaml
import os

from network import DecisionTransformer
from trainner import Trainer
from evaluation import Evaluation
from utils import SequenceDataset
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='dt')
    parser.add_argument('--env', type=str, default='pen_cloned')
    args = parser.parse_args()
    
    with open('config/default.yaml'.format(args.algo), 'r') as f:
        config = yaml.safe_load(f)
    with open('config/env/{}.yaml'.format(args.env), 'r') as f:
        config.update(yaml.safe_load(f))
    with open('config/algo/{}.yaml'.format(args.algo), 'r') as f:
        config.update(yaml.safe_load(f))
        
    if config['log_to_tensorboard']:
        path = './log/{}/{}/'.format(args.algo, args.env)
        os.makedirs(path, exist_ok=True)
        list_files = os.listdir(path)
        list_files = [int(x) for x in list_files]
        file_name = 0 if len(list_files) == 0 else max(list_files) + 1
        final_path = path+'{}'.format(file_name)
        writer = SummaryWriter(final_path)
        with open(final_path+'/config.txt', 'w') as f:
            yaml.dump(config, f)
        f.close()
    else:
        writer = None

    dataset = SequenceDataset(config)
    model = DecisionTransformer(config).to(config['device'])
        
    evaluation = Evaluation(config, state_mean=dataset.state_mean, state_std=dataset.state_std)
        
    warmup_steps = config['warmup_steps']
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/warmup_steps, 1))
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        batch_size=config['batch_size'],
        dataset=dataset,
        scheduler=scheduler,
        config=config,
        eval_fns=[evaluation.eval_fn(tar) for tar in config['env_targets']],
        writer=writer
    )
        
    for iter in range(config['max_iters']):
        outputs = trainer.train_iteration(num_steps=config['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if config['log_to_tensorboard']:
            for k, v in outputs.items():
                writer.add_scalar(k, v, iter)
                
    if config['save_model']:
        save_path = './model'
        os.makedirs(save_path, exist_ok=True)
        torch.save(model, save_path+'/{}_{}_{}.pkl'.format(args.algo, args.env, np.random.randint(10000)))

