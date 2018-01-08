require 'torch'

local count = torch.FloatTensor(10000000)

--[[
torch.manualSeed(1)
for i=1, 10000000 do
    v = torch.random(1,4)
    count[i] = v
end

print(torch.var(count))
]]

local classes={}
torch.manualSeed(1)
for i=1, 10000000 do
    v = torch.uniform()
    count[i] = v
    idx = string.format('%d',v*10)
    num = classes[idx]
    if num then
        classes[idx]  = num + 1
    else
        classes[idx] = 1
    end
end

print(torch.var(count))
for i=0, 9 do
    print(i, classes[string.format('%d',i)])
end
