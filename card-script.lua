function findPurgeBag()
    for _, obj in ipairs(getAllObjects()) do
        if obj.getName() == "Purge Bag" then
            return obj
        end
    end
    return nil
end

function purgeCard(player_color, position, object)
    local purgeBag = findPurgeBag()

    if not purgeBag then
        print("Purge Bag not found")
        return
    end

    if not object then
        print("Clicked object was nil")
        return
    end

    purgeBag.putObject(object)
end

function onLoad()
    self.addContextMenuItem("Purge", purgeCard)
end