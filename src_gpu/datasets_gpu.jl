module datasets_gpu
using DataFrames, CSV
using Random
using TimerOutputs: @timeit, get_timer

export partitionTrainTest, loadDataset

function partitionTrainTest(data, at, n, random_seeds)
    # n = nrow(data)
    Random.seed!(random_seeds)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
end

function loadDataset(dataset, run, dir_path)
    data_path = dir_path*string(dataset)*"_"*string(run)*"_"
    train = DataFrame(CSV.File(data_path*"train", header=false, missingstring="?"))
    train = Matrix(train)
    val = DataFrame(CSV.File(data_path*"val", header=false, missingstring="?"))
    val = Matrix(val)
    test = DataFrame(CSV.File(data_path*"test", header=false, missingstring="?"))
    test = Matrix(test)
    return train, val, test
end

end # end of module