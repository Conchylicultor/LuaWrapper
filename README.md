# LuaWrapper

C++ API for easy communication with Lua/Torch. Allow to launch torch networks and Lua scripts from C++ code. See `lua_wrap.hpp` and `tensor_base.hpp` for the API and `main.cpp` for some examples.

Contrary to some other existing higher level libraries, where the Lua calls are presented as string and executed as Lua code, here the functions directly call the Lua C API, so there is no compromise on performance. The drawback is that the user still has sometimes to deals with the Lua stack which offers him more control, but also more responsibilities.

To build, as usual:

```bash
mkdir build/
cd build
cmake ..
make
```
