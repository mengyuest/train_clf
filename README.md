# Learn Neural Lyapunov and Controller for Pendulum

**The simplest tryout is to just run the command in the "Train Controller" section**

## Train reference controller
(inside `rl` directory)
```
python ddpg.py
```

## Train Lyapunov
```
python train.py --exp_name clf --batch_size 5000 --sim_len 1 --lr 0.001 --use_cuda --gpus 0 --random_sampled --sample_only_once --preset_actor  
```

## Train Controller
```
python train.py --exp_name actor --batch_size 5000 --sim_len 1 --lr 0.00001 --use_cuda --gpus 0  --random_sampled --sample_only_once --preset_clf --clf_pretrained_path models/clf_36000.ckpt --eval_render
```
The pendulum should be able to swing to upright after 16k iterations


Note: right now the whole pipeline is still very sensitive to hyperparameters