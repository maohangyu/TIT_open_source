# Transformer in Transformer as Backbone for Deep Reinforcement Learning



## Overview
This is the official implementation of Transformer in Transformer (TIT) for Deep Reinforcement Learning.  
Contains scripts to reproduce experiments of the online-RL (i.e., PPO and PPO_TIT) and offline-RL (i.e., CQL and CQL_TIT).



## Result Reproduction
You can run ```bash run.sh``` to do hyperparameter search and result reproduction. 
In general, you can produce the results of offline-RL (i.e., CQL and CQL_TIT) and 
offline-SL (i.e., Decision Transformer and DT_TIT) easily.

However, to reproduce the results of online-RL (i.e., PPO and PPO_TIT), you should 
search your own hyperparameters for your own environment. Because we found that the 
performance of  online-RL is highly dependent on the evaluation environment (e.g., 
what kind of GPU).  You can also verify this by running the ```check_reproduction_of_optimization()```
function in hypertuning.py. 

Note that, although PPO_TIT need a few effort to search the hyperparameters for good 
results, it doesn't need complex optimization skills. Moreover, for CQL_TIT and DT_TIT,
we can get good results even didn't search the hyperparameters.



## Network Architecture
We try to implement the network architectures of baseline methods as closely as possible 
to their original papers and open-source code repositories. If you have more correct 
implementations, we are happy to redo the experiments based on yours.

Our TIT architecture can be found in the paper.



## Cite
Please cite our paper as:
```
@article{mao2022TIT,
  title={Transformer in Transformer as Backbone for Deep Reinforcement Learning},
  author={Mao, Hangyu and Zhao, Rui and Chen, Hao and Hao, Jianye and Chen, Yiqun and Li, Dong and Zhang, Junge and Xiao, Zhen},
  journal={arXiv preprint arXiv:2212.14538},
  year={2022}
}
```



## License

MIT
