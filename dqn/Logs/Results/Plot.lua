require 'gnuplot'

n = 0
local file1 = io.open("./Results/reward.txt")
if file1 then
    for line in file1:lines() do
	n = n+1
    end
end
file1:close()

local file1 = io.open("./Results/reward.txt")
local file2 = io.open("./Results/TD.txt")
local file3 = io.open("./Results/vavg.txt")
local file4 = io.open("./Results/goals.txt")
i = 1


x = torch.Tensor(n)
if file1 then
    for line in file1:lines() do
        x[i] = tonumber(line)
	i = i+1
    end
end

i = 1
y = torch.Tensor(n)
if file2 then
    for line in file2:lines() do
        y[i] = tonumber(line)
	i = i+1
    end
end

i = 1
z = torch.Tensor(n)
if file3 then
    for line in file3:lines() do
        z[i] = tonumber(line)
	i = i+1
    end
end

i = 1
w = torch.Tensor(n)
local maxw = 0
if file4 then
    for line in file4:lines() do
        local g, t = line:match("(%d+) (%d+)")
        local goals, trials = tonumber(g), tonumber(t)
        w[i] = goals / trials * 100
        if w[i] > maxw then
          maxw = w[i]
        end
        i = i+1
    end
end

gnuplot.pngfigure('reward.png')
gnuplot.title('reward over testing')
gnuplot.plot(x)
gnuplot.plotflush()

gnuplot.pngfigure('vavg.png')
gnuplot.title('vavg over testing')
gnuplot.plot(y)
gnuplot.plotflush()

gnuplot.pngfigure('TD_Error.png')
gnuplot.title('TD error over testing')
gnuplot.plot(z)
gnuplot.plotflush()

gnuplot.pngfigure('goals.png')
gnuplot.title('Percentage of goals / trials per epoch')
gnuplot.plot(w)
gnuplot.plotflush()
print("Maximal percentage " .. maxw)
