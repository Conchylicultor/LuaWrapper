#include <iostream>
#include <chrono>
#include <thread>

#include "lua_wrap.hpp"

using namespace std;


/** Test some of the functions of the API
  */
void unit_test(lua_State* L)
{
    (void)L;
}


/** Sample code for the wrapper
  */
int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    cout << "Lua/torch wrapper test" << endl;

    lua_State* L = LuaWrap::init_torch_vm();
    int model_nin = LuaWrap::load_model(L, "/home/teradeep/etienne/LuaWrapper/model_185_bnabs.net");
    LuaWrap::call_lua_method(L, model_nin, "evaluate"); // model:evaluate()

    for(int i = 0 ; i < 100000 ; ++i)
    {
        //THByteTensor* inputB = LuaWrap::THByte_create_tensor(1096, 1096);
        //LuaWrap::THByte_push_tensor(L, inputB);

        THFloatTensor* input = LuaWrap::THFloat_create_tensor(96, 96);
        LuaWrap::THFloat_push_tensor(L, input);

        luaT_stackdump(L);

        LuaWrap::call_lua_method(L, model_nin, "forward", 1, 1); // model:forward(input)

        if(i % 1000 == 0)
        {
            lua_gc(L, LUA_GCCOLLECT, 0);
        }

        luaT_stackdump(L);
        lua_pop(L, 1);

        std::cout << i << " " << lua_gc (L, LUA_GCCOUNT, 0) << " " << lua_gc (L, LUA_GCCOUNTB, 0) << std::endl;
    }

    cout << "The End" << endl;

    return 0;
}
