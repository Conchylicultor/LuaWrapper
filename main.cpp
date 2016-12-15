#include <iostream>
#include <chrono>
#include <thread>

#include <unistd.h>  // For working directory manipulation

#include "lua_wrap.hpp"

using namespace std;



/** Simple example
  */
void test_simple_example()
{
    // Initialise the torch virtual machine
    LuaWrap::TorchVM torchVm{};

    // Load a lua script and launch some functions
    int demo_module = torchVm.load_script("../demo_script.lua");  // demo_module = require('demo_script')
    torchVm.call_lua_method(demo_module, "foo"); // demo_module.foo()    (Static method)
    torchVm.call_lua_method(demo_module, "foo2", 0, 0, true); // demo_module:foo()    (Member)
    torchVm.call_lua_method(demo_module, "foo2", 0, 0, true);
    torchVm.call_lua_method(demo_module, "foo2", 0, 0, true);
    torchVm.call_lua_method(LUA_NOREF, "bar"); // bar()    (Global function)

    // Directly load a model and run the forward pass
    int model = torchVm.load_model("../model.net"); // model = torch.load('../model.net')
    torchVm.push_ref(model);
    torchVm.call_lua_method(LUA_NOREF, "print", 1); // print(model)
    THFloatTensor* input = torchVm.THFloat_create_tensor3d(1, 3, 3);  // Tune input dim accordingly the network
    torchVm.THFloatTensor_print(input);
    torchVm.THFloatTensor_push(input);
    torchVm.call_lua_method(model, "forward", 1, 1, true); // output = model:forward(input)
    THFloatTensor* output = torchVm.THFloatTensor_pop();
    torchVm.THFloatTensor_print(output); // print(output)
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

        luaT_stackdump(torchVm.getL());  // Print the stack (debugging)

        torchVm.call_lua_method(model, "forward", 1, 1, true); // output = model:forward(input)

        // If the garbadge collector isn't called form time to time, the memory will keep growing
        // Calling it every iterations will strongly affect the performances
        if(i % 1000 == 0)
        {
            torchVm.gc();  // Call the garbadge collector
        }

        luaT_stackdump(torchVm.getL());  // Print the stack (debugging)
        lua_pop(torchVm.getL(), 1);  // Remove the ouput

        std::cout << i << " " << lua_gc (torchVm.getL(), LUA_GCCOUNT, 0) << " " << lua_gc (torchVm.getL(), LUA_GCCOUNTB, 0) << std::endl;
    }
}


/** Calling forward through a lua script
  */
void test_memleak_script()
{
    // Set the working directories
    char* workdir = realpath("../../multipath/", NULL);
    string workdir_lua = workdir;
    free(workdir);
    if(chdir(workdir_lua.c_str()) != 0)  // We assume that the dir names won't change while the program is running
    {
        std::cerr << "Runtime Error: chdir failed" << std::endl;
        return; // We assume that the dir names won't change while the program is running
    }
    std::cout << "Workdir: " << workdir_lua << std::endl;

    LuaWrap::TorchVM torchVm{};
    int segm_module = torchVm.load_script("segmentation_main.lua");

    torchVm.call_lua_method(segm_module, "load", 0, 0, true); // model:load()


    THByteTensor *masks; // [nb_instance, height, width]
    std::vector<float> probabilities;
    std::vector<int> classes_ids;
    std::vector<std::string> classes;

    for(int i = 0 ; i < 100000 ; ++i)
    {
        std::cout << i << std::endl;
        THFloatTensor* input = torchVm.THFloat_create_tensor3d(3, 96, 96);
        THFloatStorage_fill(THFloatTensor_storage(input), 0.3*(i%2) + 0.01);  // The tensor need to be "correctly" initialized to avoid runtime crash
        torchVm.THFloatTensor_push(input);

        torchVm.call_lua_method(segm_module, "forward", 1, 4, true); // categories, classes, probs, masks = model:forward(input)

        cout << "Returned values:" << endl;
        luaT_stackdump(torchVm.getL());

        //      stack [..., self.dataset.categories, classes, probabilities, masks]
        masks = torchVm.THByteTensor_pop();  // TODO: Check if tensor valid ??
        torchVm.THByteTensor_print(masks);
        //      stack [..., self.dataset.categories, classes, probabilities]
        torchVm.pop_lua_array<float>(probabilities, LuaWrap::populate_number<float>);
        //      stack [..., self.dataset.categories, classes]
        torchVm.pop_lua_array<int>(classes_ids, LuaWrap::populate_number<int>);
        //      stack [..., self.dataset.categories]
        torchVm.pop_lua_array<std::string>(classes, &LuaWrap::populate_string);
        //      stack [...]

        // GC Don't seems necessary here
    }
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
