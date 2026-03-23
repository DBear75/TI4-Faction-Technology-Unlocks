function getHelperClient(helperObjectName)
    local function getHelperObject()
        for _, object in ipairs(getAllObjects()) do
            if object.getName() == helperObjectName then return object end
        end
        error('Missing object "' .. helperObjectName .. '"')
    end

    local helperObject = false
    local function getCallWrapper(functionName)
        helperObject = helperObject or getHelperObject()
        if not helperObject.getVar(functionName) then error('Missing ' .. helperObjectName .. '.' .. functionName) end
        return function(parameters) return helperObject.call(functionName, parameters) end
    end

    return setmetatable({}, { __index = function(t, k) return getCallWrapper(k) end })
end

_deckHelper = getHelperClient("TI4_DECK_HELPER")

function onLoad(saveState)
    injectCards()
end

function injectCards()
    local FactionTechnologyUnlockCards = {
__CARD_NAMES_BLOCK__
    }

    for expansion, cards in pairs(FactionTechnologyUnlockCards) do
        for cardName, cardData in pairs(cards) do
            local params = {
                ['deckName'] = 'Faction Technology Unlock Cards',
                ['cardName'] = cardName
            }

            if type(cardData) == "table" then
                for k, v in pairs(cardData) do
                    params[k] = v
                end
            end

            _deckHelper.injectCard(params)
        end
    end

end

local _lockGlobalsMetaTable = {}
function _lockGlobalsMetaTable.__index(table, key)
    error('Accessing missing global "' .. tostring(key or '<nil>') .. '", typo?', 2)
end
function _lockGlobalsMetaTable.__newindex(table, key, value)
    error('Globals are locked, cannot create global variable "' .. tostring(key or '<nil>') .. '"', 2)
end
setmetatable(_G, _lockGlobalsMetaTable)