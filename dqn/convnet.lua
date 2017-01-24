--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "initenv"

function create_network(args)

    local net = nn.Sequential()
    net:add(nn.Reshape(unpack(args.input_dims)))

    --- first convolutional layer
    local convLayer = nn.SpatialConvolution

    net:add(convLayer(args.hist_len*args.ncols, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],1))
    net:add(args.nl())

    -- Add convolutional layers
    for i=1,(#args.n_units-1) do
        -- second convolutional layer
        net:add(convLayer(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1]))
        net:add(args.nl())
    end

    local nel
    if args.gpu >= 0 then
        nel = net:cuda():forward(torch.zeros(1,unpack(args.input_dims))
                :cuda()):nElement()
    else
        nel = net:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
    end

    -- reshape all feature planes into a vector per example
    net:add(nn.Reshape(nel))

    -- -- fully connected layer
    -- net:add(nn.Linear(nel, args.n_hid[1]))
    -- net:add(args.nl())
    -- local last_layer_size = args.n_hid[1]
    --
    -- for i=1,(#args.n_hid-1) do
    --     -- add Linear layer
    --     last_layer_size = args.n_hid[i+1]
    --     net:add(nn.Linear(args.n_hid[i], last_layer_size))
    --     net:add(args.nl())
    -- end
    -- -- add the last fully connected layer (to actions)
    -- net:add(nn.Linear(last_layer_size, args.n_actions))

    -- Dueling Networks Architecture ------------------------------
    -- Value approximator V^(s)
    local valStream = nn.Sequential()
    valStream:add(nn.Linear(nel, args.n_hid[1]))
    valStream:add(nn.ReLU(true))
    valStream:add(nn.Linear(args.n_hid[1], 1)) -- Predicts value for state

    -- Advantage approximator A^(s, a)
    local advStream = nn.Sequential()
    advStream:add(nn.Linear(nel, args.n_hid[1]))
    advStream:add(nn.ReLU(true))
    advStream:add(nn.Linear(args.n_hid[1], args.n_actions)) -- Predicts action-conditional advantage

    -- Streams container
    local streams = nn.ConcatTable()
    streams:add(valStream)
    streams:add(advStream)

    -- Add dueling streams
    net:add(streams)
    -- Add dueling streams aggregator module
    net:add(DuelAggregator(args.n_actions))
    -- end dueling network architecture --------------------------



    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end
    return net
end

-- Creates aggregator module for a dueling architecture based on a number of discrete actions
function DuelAggregator(m)
  local aggregator = nn.Sequential()
  local aggParallel = nn.ParallelTable()

  -- Advantage duplicator (for calculating and subtracting mean)
  local advDuplicator = nn.Sequential()
  local advConcat = nn.ConcatTable()
  advConcat:add(nn.Identity())
  -- Advantage mean duplicator
  local advMeanDuplicator = nn.Sequential()
  advMeanDuplicator:add(nn.Mean(1, 1))
  advMeanDuplicator:add(nn.Replicate(m, 2, 2))
  advConcat:add(advMeanDuplicator)
  advDuplicator:add(advConcat)
  -- Subtract mean from advantage values
  advDuplicator:add(nn.CSubTable())

  -- Add value and advantage duplicators
  aggParallel:add(nn.Replicate(m, 2, 2))
  aggParallel:add(advDuplicator)

  -- Calculate Q^ = V^ + A^
  aggregator:add(aggParallel)
  aggregator:add(nn.CAddTable())

  return aggregator
end
