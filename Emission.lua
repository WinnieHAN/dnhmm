local model, parent = torch.class('nn.EmiNet', 'nn.Module')
--require 'nn'
require 'rnn'
require 'modules/Gaussian'

debug_ori = 0

function model:__init(nobs, nvars, hidsize)
    local K, V, H = nvars, nobs, hidsize  -- K: #tags V: #words
    self.K = K
    self.V= V
    self._input = torch.Tensor()--torch.range(1, K)

    if (debug_ori==1)  then
        self.net = nn.Sequential()
        self.net:add(nn.LookupTable(K, H))
        self.net:add(nn.ReLU())
        self.net:add(nn.Linear(H, H))
        self.net:add(nn.ReLU())
        self.net:add(nn.Linear(H, V))
        self.net:add(nn.LogSoftMax())
    else
        local outputsize_lstm = 5
        local s_dim = 5
        local mlp = nn.Sequential()

        local para = nn.ParallelTable()

        local tagemb = nn.Sequential()
        tagemb:add(nn.LookupTable(K, H))
        tagemb:add(nn.ReLU())
        tagemb:add(nn.View(-1,K,H))

        para:add(tagemb)

        local lstm = nn.Sequential()
        lstm:add(nn.LookupTable(V, H))
        lstm:add(nn.ReLU())
        local seqlstm = nn.SeqLSTM(H, outputsize_lstm)
        seqlstm.batchfirst = true
        lstm:add(seqlstm)
        lstm:add(nn.Select(2,-1))  -- N*outputsize_lstm
        lstm:add(nn.ReLU(true))
        local zLayer = nn.ConcatTable()
        zLayer:add(nn.Linear(outputsize_lstm, s_dim)) -- Mean μ of Z
        zLayer:add(nn.Linear(outputsize_lstm, s_dim)) -- Log variance σ^2 of Z (diagonal covariance)
        lstm:add(zLayer) -- Add Z parameter layer

        -- Create σε module
        local noiseModule = nn.Sequential()
        local noiseModuleInternal = nn.ConcatTable()
        local stdModule = nn.Sequential()
        stdModule:add(nn.MulConstant(0.5)) -- Compute 1/2 log σ^2 = log σ
        stdModule:add(nn.Exp()) -- Compute σ
        noiseModuleInternal:add(stdModule) -- Standard deviation σ
        noiseModuleInternal:add(nn.Gaussian(0, 1)) -- Sample noise ε ~ N(0, 1)
        noiseModule:add(noiseModuleInternal)
        noiseModule:add(nn.CMulTable()) -- Compute σε
        -- Create sampler q(z) = N(z; μ, σI) = μ + σε (reparametrization trick)
        local sampler = nn.Sequential()
        local samplerInternal = nn.ParallelTable()
        samplerInternal:add(nn.Identity()) -- Pass through μ
        samplerInternal:add(noiseModule) -- Create noise σ * ε
        sampler:add(samplerInternal)
        sampler:add(nn.CAddTable())
        -- ---------------

        local lstm_rep = nn.Sequential()
        lstm_rep:add(lstm)
        lstm_rep:add(sampler)
        lstm_rep:add(nn.Replicate(K,2))

        para:add(lstm_rep)

        mlp:add(para)
        mlp:add(nn.JoinTable(3))
        mlp:add(nn.Bottle(nn.Linear(H+outputsize_lstm, H)))
--        mlp:add(nn.Dropout())
        mlp:add(nn.Bottle(nn.ReLU()))
        mlp:add(nn.Bottle(nn.Linear(H, V)))
--        mlp:add(nn.Dropout())
        mlp:add(nn.Bottle(nn.LogSoftMax()))
        mlp:add(nn.View(-1,V))

        self.net = mlp
        self.encoder = lstm_rep
        self.deb = mlp --debug
    end


    self.gradOutput = torch.Tensor()---torch.Tensor(8*K, V)
    self._buffer =  torch.Tensor()
end

function model:reset()
    self.net:reset()
end

function model:parameters()
    return self.net:parameters()
end

function model:precompute()
    self._cache = self.net:forward(self._input)  -- TODO
end

function model:log_prob(input)
    local N, T = input:size(1), input:size(2)
    if (debug_ori==1)  then
        if not self._cache then
            self._logp = self.net:forward(self._input)
        else
            self._logp = self._cache --45,35534
        end
        local out_s = self._logp:index(2, input:view(-1)):view(-1, N, T):transpose(1, 2):transpose(2, 3)
--        print(out_s:size())  --256, 20, 45
--        print(input:size())  --256, 20
        return out_s
    else
--        local K = 45 -- define
        local tags = torch.range(1,self.K)
        self._input = tags:view(1,-1):expand(N,self.K):contiguous():view(-1,self.K)
--        print(self.encoder:forward(input):size())
        self._logp = self.net:forward{self._input, input} --:view(N*T, -1):index(2, input:view(-1)):view(N, T, -1) 256,45,355535
--        print('_logp initial')
--        print(self._logp:size())
--        self._logp = self._logp:view(N,self.K,-1):index(3,torch.range(1,T):long()):transpose(2, 3) -- wrong
        self._logp = torch.expand(self._logp:view(N,1,self.K,-1), N,T,self.K,self.V) --:contiguous():view(N*T,self.K,self.V)
--        print('_logp before gather')
--        print(self._logp:size())  --152 45 , 35535
        local indexs = torch.expand(input:view(N,T,1,1), N,T,self.K,1)

        self._logp = self._logp:gather(4, indexs):view(N,T,self.K)
--        print('_logp after gather')
--        print(self._logp:size()) --256 20 45
        return self._logp --:type(torch.CudaTensor)
    end
end

function model:update(input, gradOutput)
--    print('gradOutput  ')
--    print(gradOutput:size()) -- 256,20,45
    local K,V=self.K, self.V
    local N, T = input:size(1), input:size(2)
    local dx = gradOutput:transpose(2, 3):transpose(1, 2)
    self._buffer:resizeAs(dx):copy(dx)
    self.gradOutput = torch.CudaTensor(N*K,V)
    self.gradOutput:zero()
--    print('input:  ')
--    print(input:size())
--    print('self._buffer:view(-1, N * T): ')
--    print(self._buffer:view(-1, N * T):size())
    self.gradOutput:indexAdd(2, input:view(-1), self._buffer:view(-1, N * T))
--    print('self.gradOutput  ')
--    print(self.gradOutput:size())
--    print(self.gradOutput:type()) --45 35535
    self.net:backward({self._input, input}, self.gradOutput)
end


function model:parameters()
    return self.net:parameters()
end
