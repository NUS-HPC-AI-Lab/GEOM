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
```

## Condensation
To get the condensed graph.

(1) Get the initialization with coreset methods.

For example, run the following command:

```
```

(2) Optimize the condensed graph.

For example, run the following command:

```
```

## Evaluation
To evaluate the condensed graph.

For example, run the following command:
```
```


## Acknowledgement
Our code is built upon [SFGC](https://github.com/Amanda-Zheng/SFGC) and [CLNode](https://github.com/wxwmd/CLNode).

## Citation
Welcome to discuss with [yuchenzhang@std.uestc.edu.cn](mailto:yuchenzhang@std.uestc.edu.cn). If you find this repo to be useful, please cite our paper. 

```
@inproceedings{
}
```
