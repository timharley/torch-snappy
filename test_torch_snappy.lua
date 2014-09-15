require 'totem'
local snappy = require 'torch-snappy'

local tester = totem.Tester()
local tests = {}


local data_types = {
  "Byte",
  "Char",
  "Int",
  "Long",
  "Float",
  "Double",
}

for _, dtype in ipairs(data_types) do
  local tensortypename = dtype.."Tensor"
  local tensortype = torch[tensortypename]

  tests["test_"..tensortypename] = function()
    local src = tensortype.new(10,1000,1000):bernoulli(0.5)
    local bytes, closure = snappy.compress(src)
    local dst = tensortype.new():resizeAs(src):zero()
    snappy.decompress(bytes, dst)
    tester:assertTensorEq(dst, src, 0, "Compression doesn't round trip.")
    local decompressed = closure()
    tester:assertTensorEq(decompressed, src, 0, "Decompression via closure failed.")
  end

  tests["test_uncontiguous_src_"..tensortypename] = function()
    local src = tensortype.new(1000,1000):bernoulli(0.5):t()
    tester:assertError(
        function() snappy.compress(src) end,
        "Compressing a non-contiguous Tensor should error."
    )
  end

  tests["test_uncontiguous_dest"..tensortypename] = function()
    local src = tensortype.new(1000,1000):bernoulli(0.5)
    local bytes = snappy.compress(src)
    local dst = tensortype.new():resizeAs(src):zero():t()
    tester:assertError(
        function() snappy.decompress(bytes, dst) end,
        "Decompressing into a non-contiguous Tensor should error."
    )
  end

  tests["test_wrong_size_dest"..tensortypename] = function()
    local src = tensortype.new(1000,1000):bernoulli(0.5)
    local bytes = snappy.compress(src)
    local dst = tensortype.new(500,1000):zero()
    tester:assertError(
        function() snappy.decompress(bytes, dst) end,
        "Decompressing into a Tensor of wrong size should error."
    )
  end
end

tester:add(tests)
tester:run()
