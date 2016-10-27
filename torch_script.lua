-- DeepMask + MultiPathNet demo

local lfs = require 'lfs'

-- Dependencies
require 'deepmask.SharpMask'
require 'deepmask.SpatialSymmetricPadding'
require 'deepmask.InferSharpMask'
require 'inn'
require 'fbcoco'
require 'image'
local model_utils = require 'models.model_utils'
local utils = require 'utils'
local coco = require 'coco'

local dd_module = {}

function dd_module:load()
    print('--- Inside lua script (init) ---')
    print('Workdir: '..lfs.currentdir())

    -- Use fast convolution. If RAM is an issue, comment those two lines (doesn't seems to work)...
    -- It seems that the first time the forward pass is called, it'll take much more times (about x2), probably for the memory to be allocated
    --cudnn.benchmark = true
    --cudnn.fastest = true

    -- Options
    self.config = {}
    self.config.np = 60 -- number of proposals to save in test
    self.config.si = -2.5 --initial scale
    self.config.sf = .5 --final scale
    self.config.ss = .5 --scale step
    self.config.dm = false --use DeepMask version of SharpMask
    self.config.thr = 0.5 --multipathnet score threshold [0,1]
    self.config.maxsize = 600 --resize image dimension
    self.config.sharpmask_path = './data/models/sharpmask.t7' --path to sharpmask
    self.config.multipath_path = './data/models/resnet18_integral_coco.t7' --path to multipathnet

    -- Loading the models

    self.sharpmask = torch.load(self.config.sharpmask_path).model
    self.sharpmask:inference(self.config.np)

    local multipathnet = torch.load(self.config.multipath_path)
    multipathnet:evaluate()
    multipathnet:cuda()
    model_utils.testModel(multipathnet)

    self.detector = fbcoco.ImageDetect(multipathnet, model_utils.ImagenetTransformer())

    self.dataset = paths.dofile'./DataSetJSON.lua':create'coco_val2014'
    -- TODO: Could return the dataset.dataset.categories labels here

    print('--- Models loaded ---')
end

function dd_module:forward(tensor_in)
    print('--- Inside lua script (forward) ---')
    print('Workdir: '..lfs.currentdir())
    local start_time = os.clock()
    ------------------- Run DeepMask --------------------

    local meanstd = {mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 }}

    local scales = {}
    for i = self.config.si, self.config.sf, self.config.ss do
       table.insert(scales,2^i)
    end
    --for i = 1, #scales do
    --    print('Scale '..i..': '..scales[i])
   -- end

    local infer = Infer{
        np = self.config.np,
        scales = scales,
        meanstd = meanstd,
        model = self.sharpmask,
        dm = self.config.dm,
    }

    local img = tensor_in
    img = image.scale(img, self.config.maxsize)
    local h,w = img:size(2),img:size(3)

    local start_time_infer = os.clock()
    infer:forward(img:double()) -- Warning: It seems the network only accept torch.DoubleTensor
    print('Time sharpmask: '..string.format("%.4f",  os.clock() - start_time_infer))

    local masks,_ = infer:getTopProps(.2,h,w)

    local Rs = coco.MaskApi.encode(masks)
    local bboxes = coco.MaskApi.toBbox(Rs)
    bboxes:narrow(2,3,2):add(bboxes:narrow(2,1,2)) -- convert from x,y,w,h to x1,y1,x2,y2

    ------------------- Run MultiPathNet --------------------

    local start_time_infer = os.clock()
    local detections = self.detector:detect(img:float(), bboxes:float())
    local prob, maxes = detections:max(2)
    print('Time multipath: '..string.format("%.4f",  os.clock() - start_time_infer))

    -- remove background detections
    local idx = maxes:squeeze():gt(1):cmul(prob:gt(self.config.thr)):nonzero()
    if idx:nDimension() == 0 then -- Premature exit
        return self.dataset.dataset.categories, {}, {}, torch.ByteTensor(1,h,w):zero()
    end

    idx = idx:select(2,1)
    bboxes = bboxes:index(1, idx)
    maxes = maxes:index(1, idx)
    prob = prob:index(1, idx)

    local scored_boxes = torch.cat(bboxes:float(), prob:float(), 2)
    local final_idx = utils.nms_dense(scored_boxes, 0.3)

    --print('---- scored_boxes ----')
    --print(scored_boxes:type())
    --print(scored_boxes:size())
    --print(scored_boxes)
    --print('---- final_idx ----')
    --print(final_idx:type())
    --print(final_idx:size())
    --print(final_idx)

    ------------------- Cleanup data --------------------

    -- remove suppressed masks
    masks = masks:index(1, idx):index(1, final_idx)

    local end_time = os.clock()
    local elapsed_time = end_time-start_time
    print('Time loop: '..string.format("%.4f", elapsed_time))

    --print('---- masks ----')
    --print(masks:type())
    --print(masks:size())

    --print('Nb classes:', #self.dataset.dataset.categories) -- 80 classes

    local classes = {}
    local probabilities = {}
    for i,v in ipairs(final_idx:totable()) do -- TODO: Not sure about the order of the ids (does that correspond to the masks order ?)
        classes[i] = (maxes[v][1]-1) - 1 -- We remove 1 to compensate the fact that the lua array start at 1
        probabilities[i] = prob[v][1]
    end

    if true then
        return self.dataset.dataset.categories,
            classes,
            probabilities,
            masks  -- Warning: the masks may overlapp
    end

    ------------------- Draw detections (never reached) --------------------
    local res = img:clone()
    coco.MaskApi.drawMasks(res, masks, 10)
    for i,v in ipairs(final_idx:totable()) do
        local class = maxes[v][1]-1
        local x1,y1,x2,y2 = table.unpack(bboxes[v]:totable())
        y2 = math.min(y2, res:size(2)) - 10
        local name = self.dataset.dataset.categories[class]
        print(prob[v][1], class, name)
        image.drawText(res, name, x1, y2, {bg={255,255,255}, inplace=true})
    end
    image.save(string.format('./res.jpg',self.config.model),res)

    print('| done')


    print('--- End forward ---')
    return nil -- TODO: WARNING! Should be a Tensor3d
end

return dd_module
