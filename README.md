# G3RD
PyTorch implementation for 'G3RD: Generalizable Generative Graph Representation Learning via Multi-Teacher Distillation'
## Dependencies
* torch 1.12.0+cu116 
* torch-cluster 1.6.0 
* torch-geometric 2.1.0 
* torch-scatter 2.0.9
* torch-sparse 0.6.15 

If you have trouble in installing `torch-geometric`, you may find help in its [official website](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Training & Evaluation
### Node Classification and Clustering
```
sh run.sh
```
### Link Prediction
```
sh run_lp.sh
```

## Acknowledgements
Parts of implementation are reference to [GRACE](https://github.com/CRIPAC-DIG/GRACE)
