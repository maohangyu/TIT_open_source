# Transformer in Transformer as Backbone for Deep Reinforcement Learning



## Overview
This is the official implementation of Transformer in Transformer (TIT) for Deep Reinforcement Learning.  
Contains scripts to reproduce experiments of the offline-SL (i.e., Decision Transformer and DT_TIT).



## Network Architecture

Decision Transformer (DT):  

![image info](./DT.png)

DT vs DT_TIT:  

![image info](./DT_TIT.png)



## Instructions

Our implementation is highly based on [Decision Transformer (DT)](https://github.com/kzl/decision-transformer).   

We provide code in two sub-directories:  
1. `gym` containing code for d4rl-MoJoCo experiments. Run `bash run.sh`, and it can reproduce our results shown in the paper.
2. `atari` containing code for d4rl-Atari experiments. Run `bash run.sh`, and it can run normally, but the results may be poor (our GPU was often OOM, so we didn't try Atari at all).



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


## Acknowledgements

Please cite Decision Transformer as:
```
@article{chen2021decisiontransformer,
  title={Decision Transformer: Reinforcement Learning via Sequence Modeling},
  author={Lili Chen and Kevin Lu and Aravind Rajeswaran and Kimin Lee and Aditya Grover and Michael Laskin and Pieter Abbeel and Aravind Srinivas and Igor Mordatch},
  journal={arXiv preprint arXiv:2106.01345},
  year={2021}
}
```

## License

MIT
