# Moving Horizon Classification Tree
Draft version for reviewing only, all rights reserved. Will be updated and open-sourced after the paper is published.

## Requirements
* CUDA v11.4
* Julia
  * AbstractTrees v0.4.4
  * CSV v0.10.9
  * CUDA v4.0.1
  * CUDAKernels v0.4.7
  * CategoricalArrays v0.10.7
  * DataFrames v1.5.0
  * DecisionTree v0.12.3
  * Distances v0.10.7
  * Distributions v0.25.81
  * KernelAbstractions v0.8.6
  * MLDataUtils v0.5.4
  * MathOptInterface v1.12.0
  * Metaheuristics v3.2.14
  * Metrics v0.1.2
  * ScikitLearn v0.6.6
  * ScikitLearnBase v0.5.0
  * StatsBase v0.33.21
  * StatsPlots v0.15.4
  * TimerOutputs v0.5.22
  * Tullio v0.3.5
  * InteractiveUtils
  * LinearAlgebra
  * Printf
  * Random
  * Statistics

## File list
###  Source Files
* ./src_gpu/ - GPU accelerated moving horizon classification tree.
  * datasets_gpu.jl - load dataset.
  * warmmstart_gpu.jl - CART warm start for GPU accelerated moving horizon classification tree.
  * oct_gpu.jl - GPU accelerated fitness evaluation functions.
  * de_gpu.jl - GPU accelerated differential evolution and moving horizon algorithm.
* ./src_cart/ - adapted from DecisionTree.jl and add misclassification error as the fitness function.
  * used in the deepest branch node level of the moving horizon classification tree.
* ./src/ - local search classification tree.
  * datasets.jl - load dataset.
  * warmmstart.jl - CART warm start for local search classification tree.
  * oct.jl - fitness evaluation functions.
  * local_search.jl - self-implemented local search algorithm from Bertsimas and Dunn, 2017.
  
### Test Files - ./test/
  * test_cart_de_la_gpu.jl - test the GPU accelerated moving horizon classification tree.
  * test_local_search.jl - test the local search classification tree.
  * test_alpha_tuning.jl - alpha tuning for the GPU accelerated moving horizon classification tree.
  * test_alpha_tuning_cart.jl - alpha tuning for CART.

## Evaluation
Please refer to test_*.jl file for the meaning of each input parameter 
* Dataset List
    * We include the used UCI dataset in the ./data/ folder. Dataset number from 1 to 68, each dataset is randomly splitted 10 times in the proportion of 75% training and 25% testing (without tuning) and 50% training, 25% validation and 25% testing (with tuning).
    * The detailed information of each dataset is in the ./data/dataset_info.pdf file. Due to the size limit, we include 10 splits for dataset 1-64 and the first split for the dataset 65 with one million samples. For the datasets larger than 1 million samples, you can find the link in the ./data/dataset_info.pdf file.
  
* Examples for MH-DEOCT without alpha tuning
```shell
# D2P2, dataset 1-65, run 1-10
julia test/test_cart_de_la_gpu.jl 1 65 2 OCT_az 0 10 2 1 rlt/ 0 100 600 1 0 1 
# D3P3, dataset 1-65, run 1-10
julia test/test_cart_de_la_gpu.jl 1 65 3 OCT_az 0 10 3 1 rlt/ 0 100 600 1 0 1
```

* Examples for others
```shell
# MH-DEOCT with alpha tuning, D2P2, dataset 1-65, run 1-10
julia test/test_alpha_tuning.jl 1 65 2 OCT_az 0 10 2 1 rlt/ 0 100 600 1 0 1
# CART with alpha tuning, D2, dataset 1-65, run 1-10
julia test/test_alpha_tuning_cart.jl 1 65 2 OCT_az 0 10 2 1 rlt/ 0 100 600 1 0 1
# Local search, D2, dataset 1-65, run 1-10, 100 initial solutions
julia test/test_local_search.jl 1 65 2 1 10 100 1 rlt/ 1
```