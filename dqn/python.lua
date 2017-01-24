local path = os.getenv("LUA_SOPATH")
if path then
	func = package.loadlib(path.."/lua-python.so", "luaopen_python")
	if func then
		func()
		return
	end
end
local modmask = "/usr/local/lib/python2.7/dist-packages/lua-python.so"
local loaded = false

local func = assert(package.loadlib(modmask, "luaopen_python"))
if func then
	loaded = true
	func()
end
if not loaded then
	error("python.lua: unable to find python module")
end
