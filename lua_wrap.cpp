#include "lua_wrap.hpp"

#include <iostream>


namespace LuaWrap
{

////////////////////////// High Level API //////////////////////////

lua_State* init_torch_vm()
{
    // Initialize Lua
    lua_State *L = luaL_newstate(); //start lua VM
    if(L == NULL) {
        throw LuaException("Could not create lua_State()");
    }

    // Load standard libs
    luaL_openlibs(L);        //load standard lua libs
    load_lualib(L, "torch"); //load torch and nn
    load_lualib(L, "nn");

    // TODO: Set Default torch tensor ?
    // TODO: Set heap tracking ?

    // Calling torch.setheaptracking(false)
//    lua_getglobal(L, "torch");                                     // stack = [torch]
//    int torch_reg = luaL_ref(L,LUA_REGISTRYINDEX);                 // stack = []
//    lua_pushboolean(L, 0);                                         // stack = [false]
//    call_lua_method(L, torch_reg, "setheaptracking", 1, 0, false); // stack = []

    ASSERT_STATE(lua_gettop(L) == 0); // Final state empty

    return L;
}


int load_script(lua_State* L, const std::string& script_name)
{
    int stack_size = lua_gettop(L);

    // TODO: Send init parameters to the script (array of values ?, or values pushed on the stack by the caller)

    // Loading the module:                    Initial state:  stack = [...]
    check_error(luaL_loadfile(L, script_name.c_str()), L); // stack = [..., chunk]
    check_error(lua_pcall(L,0,1,0), L);                    // stack = [..., dd_module]
    int script_reg = luaL_ref(L,LUA_REGISTRYINDEX);        // stack = [...]
    call_lua_method(L, script_reg, "load");                // stack = [...] (calling dd_module:load())

    ASSERT_STATE(lua_gettop(L) == stack_size); // Leave the stack as we found it

    return script_reg;
}


int load_model(lua_State* L, const std::string& model_name)
{
    ASSERT_STATE(lua_gettop(L) == 0); //   Initial state:  stack = []

    // Get the torch module
    lua_getglobal(L, "torch");                          // stack = [torch]
    int torch_reg = luaL_ref(L,LUA_REGISTRYINDEX);      // stack = []

    // Call torch.load("model_name")
    lua_pushstring(L, model_name.c_str());              // stack = ["model_name"]
    call_lua_method(L, torch_reg, "load", 1, 1, false); // stack = [model] (input: model name, output: model obj)

    // Pop and save the result
    int model_reg = luaL_ref(L,LUA_REGISTRYINDEX);      // stack = []

    ASSERT_STATE(lua_gettop(L) == 0); // Final state empty

    return model_reg;
}


void load_lualib(lua_State* L, const std::string& lib_name)
{
    lua_getglobal(L, "require"); // Put the require function on the stack
    lua_pushstring(L, lib_name.c_str()); // Push lib_name to the stack
    check_error(lua_pcall(L,1,0,0), L); // Equivalent to: require(lib_name)
}


void call_lua_method(
    lua_State* L,
    int instance_ref,
    const std::string& method_name,
    int nb_in,
    int nb_out,
    bool is_class
)
{
    int stack_size = lua_gettop(L);
    ASSERT_STATE(stack_size >= nb_in); // The method arguments should have been pushed

    int offset = 0; // Take the self argument into account

    //                                     Initial state: stack = [...,<args>]
    lua_rawgeti(L, LUA_REGISTRYINDEX, instance_ref);   // stack = [...,<args>, instance]
    lua_pushstring(L, method_name.c_str());            // stack = [...,<args>, instance, "method_name"]
    lua_gettable(L, -2);                               // stack = [...,<args>, instance, instance:method()]
    if(is_class) // Push the self argument
    {
        lua_pushvalue(L, -2);                          // stack = [...,<args>, instance, instance:method(), instance]
        offset = 1; // Take the self ref into account
    }
    lua_remove(L,-2-offset);                           // stack = [...,<args>, instance:method() (, instance)]
    for (int i = 0; i < nb_in; ++i) // Add arguments in the order
    {
        int current_stack_length = nb_in + 1 + offset;
        ASSERT_STATE(lua_gettop(L) == current_stack_length - (stack_size - nb_in));
        lua_pushvalue(L, -current_stack_length);       // stack = [...,<args>, instance:method() (, instance), <args>]
        lua_remove(L, -(current_stack_length+1));      // stack = [...,instance:method()(, instance), <args>]
    }
    check_error(
                lua_pcall(L, offset+nb_in, nb_out, 0), // stack = [...,<returns>] (Function called)
                L);

    ASSERT_STATE(lua_gettop(L) == stack_size - nb_in + nb_out); // Sanity check
}


template<typename T>
void pop_lua_array(
    lua_State* L,
    std::vector<T>& out_array,
    T (*populate_fct)(lua_State*) // Read the value on top of the stack and return it [-0,+0,-]
)
{
    int stack_size = lua_gettop(L);

    out_array.clear(); // Free the previous values

    //              Initial state:  stack = [..., table]
    ASSERT_STATE(stack_size >= 1);
    ASSERT_STATE(lua_type(L, -1) == LUA_TTABLE);
    lua_pushnil(L);              // stack = [..., table, nil] (first key)
    while (lua_next(L, -2) != 0) // stack = [..., table, key, value]
    {
        out_array.push_back(populate_fct(L));
        lua_pop(L, 1);           // stack = [..., table, key]
    }
    lua_pop(L,1);                // stack = [...]

    ASSERT_STATE(lua_gettop(L) == stack_size - 1);
}


std::string populate_string(lua_State* L)
{
    ASSERT_STATE(lua_type(L, -1) == LUA_TSTRING);
    return std::string(lua_tostring(L, -1));
}

template <typename TNumber>
TNumber populate_number(lua_State* L)
{
    ASSERT_STATE(lua_type(L, -1) == LUA_TNUMBER);
    return static_cast<TNumber>(lua_tonumber(L, -1));
}


////////////////////////// OpenCv/Tensor manipulation API //////////////////////////


// Define the generics here
#include "tensor_all.cpp"


////////////////////////// Low level Level API //////////////////////////


void check_error(int returnedValue, lua_State *L)
{
    if (returnedValue)
    {
        std::string errMsg(lua_tostring(L, -1));
        // TODO: Clear the stack ?? Free memory ?? (otherwise, push above the
        // limit !!)
        throw LuaException(errMsg);
    }
}


void print_tensor(lua_State* L, THFloatTensor* tensor)
{
    lua_getglobal(L,"print");
    luaT_pushudata(L, (void*) tensor, "torch.FloatTensor");
    lua_pcall(L,1,0,0);
}


void set_defaultfloattensor(lua_State* L)
{
    int stack_size = lua_gettop(L);

    // TODO: Cloud be done with call_lua_method (Error checking)
    lua_getglobal(L, "torch");
    lua_pushstring(L, "setdefaulttensortype");
    lua_gettable(L,-2); lua_remove(L,-2);

    lua_pushstring(L, "torch.FloatTensor");
    lua_pcall(L,1,0,0);

    ASSERT_STATE(lua_gettop(L) == stack_size);
}


LuaException::LuaException(const std::string& message) : _message("Lua error: " + message)
{
}

const char* LuaException::what() const noexcept
{
    return _message.c_str();
}


} // End of namespace
