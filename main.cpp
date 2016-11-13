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


/** Loop over forward with different inputs tensors,
  * The memory should stay constant
  */
void test_memleak()
{
    auto torchVm = LuaWrap::TorchVM();
    int model_nin = torchVm.load_model("../model_185_bnabs.net");
    torchVm.call_lua_method(model_nin, "evaluate"); // model:evaluate()

    for(int i = 0 ; i < 100000 ; ++i)
    {
        THFloatTensor* input = torchVm.THFloat_create_tensor(96, 96);
        torchVm.THFloat_push_tensor(input);

        luaT_stackdump(torchVm.getL());

        torchVm.call_lua_method(model_nin, "forward", 1, 1); // model:forward(input)

        // If the garbadge collector isn't called form time to time, the memory will keep growing
        // Calling it every iterations will strongly affect the performances
        if(i % 1000 == 0)
        {
            lua_gc(torchVm.getL(), LUA_GCCOLLECT, 0);
        }

        luaT_stackdump(torchVm.getL());
        lua_pop(torchVm.getL(), 1);

        std::cout << i << " " << lua_gc (torchVm.getL(), LUA_GCCOUNT, 0) << " " << lua_gc (torchVm.getL(), LUA_GCCOUNTB, 0) << std::endl;
    }
}


/** Calling forward through a lua script
  */
void test_memleak_script()
{

}


/** Sample code for the wrapper
  */
int main(int argc, char** argv)
{
    (void)argc;
    (void)argv; // Unused

    cout << "Lua/torch wrapper test" << endl;

    test_memleak();
    test_memleak_script();

    cout << "The End" << endl;

    return 0;
}
