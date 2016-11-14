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


/** Simple example
  */
void test_simple_example()
{
    // Initialise the torch virtual machine
    LuaWrap::TorchVM torchVm{};

    // Load a lua script and launch some functions
    int demo_module = torchVm.load_script("../demo_script.lua");
    torchVm.call_lua_method(demo_module, "foo"); // Static method
    torchVm.call_lua_method(demo_module, "foo2", 0, 0, true); // Member
    torchVm.call_lua_method(demo_module, "foo2", 0, 0, true);
    torchVm.call_lua_method(demo_module, "foo2", 0, 0, true);
    torchVm.call_lua_method(LUA_NOREF, "bar"); // Global function

    // Directly load a model and run the forward pass
    int model = torchVm.load_model("../model.net");
    torchVm.push_ref(model);
    torchVm.call_lua_method(LUA_NOREF, "print", 1); // print(model)
    THFloatTensor* input = torchVm.THFloat_create_tensor3d(1, 3, 3);
    torchVm.THFloatTensor_print(input);
    torchVm.THFloatTensor_push(input);
    torchVm.call_lua_method(model, "forward", 1, 1, true); // model:forward(input)
    THFloatTensor* output = torchVm.THFloatTensor_pop();
    torchVm.THFloatTensor_print(output);
}


/** Loop over forward with different inputs tensors,
  * The memory should stay constant
  */
void test_memleak()
{
    LuaWrap::TorchVM torchVm{};
    int model = torchVm.load_model("../model.net");
    torchVm.call_lua_method(model, "evaluate", 0, 0, true); // model:evaluate()

    for(int i = 0 ; i < 100000 ; ++i)
    {
        THFloatTensor* input = torchVm.THFloat_create_tensor3d(3, 96, 96);
        torchVm.THFloatTensor_push(input);

        luaT_stackdump(torchVm.getL());

        torchVm.call_lua_method(model, "forward", 1, 1, true); // model:forward(input)

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

    test_simple_example();
    //test_memleak();
    //test_memleak_script();

    cout << "The End" << endl;

    return 0;
}
