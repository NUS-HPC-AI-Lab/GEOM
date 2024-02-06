# GEOM
Pytorch implementation of "Navigating Complexity: Toward Lossless Graph Condensation via Expanding Window Matching".

The overall framework of the proposed GEOM:

![pipeline](figures/pipeline.png)

In the buffer phase, we train the expert trajectories with curriculum learning to involve more informative supervision signals from the original graph. In the condensation phase, we utilize expanding window matching to capture the rich information. Moreover, a knowledge embedding extractor is used to further extract knowledge from the expert trajectories with a new perspective.

## Requirements
Please see [requirements](/requirements).

Run the following command to install:

```
pip install -r requirements.txt
```

## Buffer
To get expert trajectories. 

For example, run the following command:

```
CUDA_VISIBLE_DEVICES=4 python buffer_transduct_cl.py --lr_teacher 0.4 \
--teacher_epochs 3000 --dataset cora   \
--num_experts=200 --wd_teacher 0 --optim SGD --lam 0.75 --T 1500 --scheduler geom
```

## Condensation
To get the condensed graph.

(1) Get the initialization with coreset methods.

For example, run the following command:

```
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 1 --lr 0.01 --lr_coreset 0.005 \
--weight_decay 5e-4 --wd_coreset 5e-4 --save 1 --method kcenter --reduction_rate 0.5
```

(2) Optimize the condensed graph.

For example, run the following command:

```
python condense_transduct_2.py --section cora-r05  --gpuid 0 --lam 0.75 --T 1500 --scheduler geom   \
--min_start_epoch 0 --max_start_epoch 200 --expert_epochs 1400   \
--syn_steps 2500 --max_start_epoch_s 50 --lr_feat 0.0001 --lr_y 0.00005 --beta 0.01 --soft_label
```

## Evaluation
To evaluate the condensed graph.

For example, run the following command:
```
python eval_condg.py --section cora-r05 
```


## Acknowledgement
Our code is built upon [SFGC](https://github.com/Amanda-Zheng/SFGC) and [CLNode](https://github.com/wxwmd/CLNode).

## Citation
Welcome to discuss with [yuchenzhang@std.uestc.edu.cn](mailto:yuchenzhang@std.uestc.edu.cn). If you find this repo to be useful, please cite our paper. 

```
@inproceedings{
}
```
