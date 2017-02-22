
require 'image'
require 'sys'

local hfowrap = {}

-- This file defines the hfowrap.GameEnvironment class.
package.loadlib("libpython2.7.so", "*")
require 'python'

-- The GameEnvironment class.
local gameEnv = torch.class('GameEnvironment')


function gameEnv:__init(_opt)
    python.execute("import api_agent")
    assert(_opt.image_port, "opt image port is nil")
    assert(_opt.server_port, "opt server port is nil")
    if 0 == _opt.onenet then
        self.api_agent = python.eval(string.format("api_agent.dqnhfoAgent(%d, %d, %d)", _opt.server_port, _opt.image_port, _opt.teamplay))
    else
        self.api_agent = python.eval(string.format("api_agent.doubledqnhfoAgent(%d, %d)", _opt.server_port, _opt.image_port))
    end
    local _opt = _opt or {}
    self.img_buffer = ""
    -- defaults to emulator speed
    self.verbose        = _opt.verbose or 0
    self._actrep        = _opt.actrep or 1
    self._random_starts = _opt.random_starts or 1
    self:reset(_opt.env, _opt.env_params, _opt.gpu)
    return self
end


function gameEnv:_updateState(frame, reward, terminal)
    self._state.reward       = reward
    self._state.terminal     = terminal
    self._state.observation  = frame
    return self
end


function gameEnv:getState()
    -- grab the screen again only if the state has been updated in the meantime
    if not self._state.observation then
        self._state.observation = self:_getScreen()
    end
    -- why not call api_agent's getreward and getterminal functions here?

    return self._state.observation, self._state.reward, self._state.terminal
end


function gameEnv:reset(_env, _params, _gpu)
    local params = _params or {useRGB=true}
    -- if no game name given use previous name if available

    self._actions   = self:getActions()

    -- start the game
    if self.verbose > 0 then
        print('\nPlaying: HFO!')
    end

    self._state = self._state or {}
    self:_updateState(self:_step(0))
    self:getState()
    return self
end


-- Function plays `action` in the game and return game state.
function gameEnv:_step(action)
    assert(action)
    -- play step with action
    self.api_agent.act(action)
    local terminal = self.api_agent.agentStep()
    local screen = self:_getScreen()
    local reward = self.api_agent.getReward()
    return screen, reward, terminal
end


function gameEnv:_getScreen()
   ---- This is the NEW WAY ----
   local datacount = self.api_agent.height * self.api_agent.width * 3
   local stringimage = ''

   while datacount > 0 do
     self.img_buffer = self.img_buffer .. self.api_agent.img_socket.recv(datacount, 0x40)

     local buflen = string.len(self.img_buffer)
     if buflen == 0 then
       return
     end

     if datacount - buflen >= 0 then
       stringimage = stringimage .. self.img_buffer
       datacount = datacount - buflen
       self.img_buffer = ""
     else
       stringimage = stringimage .. string.sub(self.img_buffer, 1, datacount)
       self.img_buffer = string.sub(self.img_buffer, datacount + 1)
       datacount = 0
     end
   end

   local imstore = torch.ByteStorage():string(stringimage)
   local img_byte_tensor = torch.ByteTensor(imstore, 1, torch.LongStorage{252, 252, 3})
   local img_tensor = img_byte_tensor:type('torch.FloatTensor'):div(255)
   --local img_tensor = image.scale(large_img_tensor, 84, 84)
   --img_tensor:transpose(1,3)
   --print(string.format("Tensor size is %d, %d, %d", img_tensor:size(1), img_tensor:size(2), img_tensor:size(3)))
   --local win = image.display({image=img_tensor, win=win})

   ---- This is the OLD WAY ----
   -- local img_table = self.api_agent.getScreen()
   -- local img_tensor = torch.Tensor(img_table)
   return img_tensor:transpose(1,3) --:transpose(2,3):transpose(1,2)
end

-- Function plays one random action in the game and return game state.
function gameEnv:_randomStep()
    return self:_step(self._actions[torch.random(#self._actions)])
end


function gameEnv:step(action, training)
    -- accumulate rewards over actrep action repeats
    local cumulated_reward = 0
    local frame, reward, terminal
    for i=1,self._actrep do
        -- Take selected action
        -- maybe change to directly call api_agent's agentStep
        frame, reward, terminal = self:_step(action)

        -- accumulate instantaneous reward
        cumulated_reward = cumulated_reward + reward

        -- game over, no point to repeat current action
        if terminal then break end
    end
    self:_updateState(frame, cumulated_reward, terminal)
    return self:getState()
end


--[[ Function advances the emulator state until a new game starts and returns
this state. The new game may be a different one, in the sense that playing back
the exact same sequence of actions will result in different outcomes.
]]
function gameEnv:newGame()
    local obs, reward, terminal
    terminal = self._state.terminal
    while not terminal do
        obs, reward, terminal = self:_randomStep()
    end
    -- take one null action in the new game
    return self:_updateState(self:_step(0)):getState()
end


--[[ Function advances the emulator state until a new (random) game starts and
returns this state.
]]
function gameEnv:nextRandomGame(k)
    local obs, reward, terminal = self:newGame()
    k = k or torch.random(self._random_starts)
    for i=1,k-1 do
        obs, reward, terminal = self:_step(0)
        if terminal then
            print(string.format('WARNING: Terminal signal received after %d 0-steps', i))
        end
    end
    return self:_updateState(self:_step(0)):getState()
end


--[[ Function returns the number total number of pixels in one frame/observation
from the current game.
]]
function gameEnv:nObsFeature()
    -- return self.api_agent.getStateDims()
    return self.api_agent.height*self.api_agent.width
end


-- Function returns a table with valid actions in the current game.
function gameEnv:getActions()
    return self.api_agent.getActions()
end


return hfowrap
