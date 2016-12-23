# LuaWrapper

C++ API for easy communication with Lua/Torch. Allow to launch torch networks and Lua scripts from C++ code. See `lua_wrap.hpp` and `tensor_base.hpp` for the API and `main.cpp` for some examples.

Contrary to some other existing higher level libraries, where the Lua calls are presented as string and executed as Lua code, here the functions directly call the Lua C API, so there is no compromise on performance. The drawback is that the user still has sometimes to deals with the Lua stack which offers him more control, but also more responsibilities.

The program require a standard Torch installation. To build, as usual:

```bash
mkdir build/
cd build
cmake ..
make
```

Then, to use the library, just include `lua_wrap.hpp` (present in the `include/` folder) and link the generated `lib/liblua_wrap.so`. After compilation, the example present in the `build/` folder can be launched with `./a.out`.

LuaWrap can be used in two main modes:
* By directly loading and executing a Torch model (`.t7`) directly from the C++ code. No Lua script is required. This is used when there is no complex interaction between models and we just want to perform a forward pass on an image or data.
* By interfacing some Lua scripts with the C++ program. This allows more complex interactions. One use case for instance is to quickly integrate existing Torch repositories within the C++ program without having to reprogram all the data loading, image pre-processing and so on.

Of course, you can use both ways within the same program.
