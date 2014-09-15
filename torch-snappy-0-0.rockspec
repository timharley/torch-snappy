package = "torch-snappy"
version = "0-0"

source = {
   url = "git://github.com/timharley/torch-snappy.git",
   tag = "master"
}

description = {
   summary = "Snappy bindings for Torch",
   homepage = "https://github.com/timharley/torch-snappy.git",
}

dependencies = {
   "torch >= 7.0",
   "totem",
}

build = {
    type = "builtin",
    modules = {
        ['torch-snappy.init'] = 'torch-snappy/init.lua'
    }
}
