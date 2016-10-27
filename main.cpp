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
    (void)L;

    THFloatTensor* input = LuaWrap::THFloat_create_tensor(96, 96);

    cout << "The End" << endl;

    return 0;
}
