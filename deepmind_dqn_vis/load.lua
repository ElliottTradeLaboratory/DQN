require 'torch'
require 'image'

local Lua = torch.class('Lua')

function Lua:load(path)

	f = torch.DiskFile(path, 'r'):binary()
	local buf = f:readObject()
	f:close()

	return buf
end

function Lua:loadImage(path)

	return image.load(path,3,'byte')
end

function Lua:checkImage(dstimg, path)

	srcimg = self:loadImage(path)
	
	local c, w, h = srcimg:size()[1], srcimg:size()[2], srcimg:size()[3]
	
	local n_unmatch = 0
	print(dstimg:size())
	
	for i=c, 1, -1 do
		for j=w, 1, -1 do
			for k=h, 1, -1 do
				if srcimg[i][j][k] ~= dstimg[i][j][k] then
					n_unmatch = n_unmatch + 1
					print(srcimg[i][j][k] , dstimg[i][j][k])
				end
			end
		end
	end
	
	if n_unmatch > 0 then
		print('NG', n_unmatch)
	else
		print('OK')
	end
end
