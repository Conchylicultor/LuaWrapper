#include "lua_wrap.hpp"

#include <iostream>



////////////////////////// Assersions macros //////////////////////////

/** Macro which assert if the condition is valid. Mainly called to check if the lua stack
  * is in a valid state
  */
#define ASSERT_STATE(valid) \
    if (!(valid)) \
    {\
        throw LuaException("Invalid state in " + std::string(__FILE__) + " at line " + std::to_string(__LINE__) + ": " LUAW_STRINGIFY(valid));\
    }

/** If returnedValue != 0, try to capture the error from lua.
  * Raise an exception containing the error message
  * WARNING: If the exception is thrown, the stack will be left in an unknown state
  */
#define CHECK_ERROR(rcode) \
    if (rcode) \
    {\
        std::string errMsg(lua_tostring(L, -1));\
        throw LuaException("Runtime Error in " + std::string(__FILE__) + " at line " + std::to_string(__LINE__) + ": " + errMsg);\
    }


namespace LuaWrap
{

////////////////////////// High Level API //////////////////////////

TorchVM::TorchVM()
{
    // Initialize Lua
    L = luaL_newstate(); //start lua VM
    if(L == nullptr) {
        throw LuaException("Could not create lua_State()");
    }

    // Load standard libs
    luaL_openlibs(L);        //load standard lua libs
    load_lualib("torch"); //load torch and nn
    load_lualib("nn");

    // Get the torch module
    lua_getglobal(L, "torch");                // stack = [torch]
    torch = luaL_ref(L,LUA_REGISTRYINDEX);    // stack = []

    // Some configuration options
    THFloat_setdefaulttensortype();


    // TODO: Set heap tracking ?

    // Calling torch.setheaptracking(false)
//    lua_getglobal(L, "torch");                                     // stack = [torch]
//    int torch_reg = luaL_ref(L,LUA_REGISTRYINDEX);                 // stack = []
//    lua_pushboolean(L, 0);                                         // stack = [false]
//    call_lua_method(L, torch_reg, "setheaptracking", 1, 0, false); // stack = []

    ASSERT_STATE(lua_gettop(L) == 0); // Final state empty
}


TorchVM::~TorchVM()
{
    lua_close(L);
}


int TorchVM::load_script(const std::string& script_name)
{
    int stack_size = lua_gettop(L);

    // Loading the module:                 Initial state:  stack = [...]
    CHECK_ERROR(luaL_loadfile(L, script_name.c_str())); // stack = [..., chunk]
    CHECK_ERROR(lua_pcall(L,0,1,0));                    // stack = [..., dd_module]
    int script_reg = luaL_ref(L,LUA_REGISTRYINDEX);     // stack = [...]

    ASSERT_STATE(lua_gettop(L) == stack_size); // Leave the stack as we found it

    return script_reg;
}


int TorchVM::load_model(const std::string& model_name)
{
    int stack_size = lua_gettop(L);

    // Call torch.load("model_name")       Initial state:  stack = [...]
    lua_pushstring(L, model_name.c_str());              // stack = [..., "model_name"]
    call_lua_method(torch, "load", 1, 1, false);        // stack = [..., model] (input: model name, output: model obj)

    // Pop and save the result
    int model_reg = luaL_ref(L,LUA_REGISTRYINDEX);      // stack = [...]

    ASSERT_STATE(lua_gettop(L) == stack_size); // Final state empty

    return model_reg;
}


void TorchVM::load_lualib(const std::string& lib_name)
{
    lua_pushstring(L, lib_name.c_str()); // Push lib_name to the stack
    call_lua_method(LUA_NOREF, "require", 1, 0); // require(lib_name)
}


void TorchVM::call_lua_method(
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

    //                                       Initial state: stack = [...,<args>]
    if(instance_ref != LUA_NOREF)
    {
        lua_rawgeti(L, LUA_REGISTRYINDEX, instance_ref); // stack = [...,<args>, instance]
        lua_pushstring(L, method_name.c_str());          // stack = [...,<args>, instance, "method_name"]
        lua_gettable(L, -2);                             // stack = [...,<args>, instance, instance:method()]
        if(is_class) // Push the self argument
        {
            lua_pushvalue(L, -2);                        // stack = [...,<args>, instance, instance:method(), instance]
            offset = 1; // Take the self ref into account
        }
        lua_remove(L,-2-offset);                         // stack = [...,<args>, instance:method() (, instance)]
    }
    else
    {
        ASSERT_STATE(is_class == false);
        lua_getglobal(L, method_name.c_str());           // stack = [...,<args>, fct()]
    }
    for (int i = 0; i < nb_in; ++i) // Add arguments in the order
    {
        ASSERT_STATE(lua_gettop(L) == stack_size + 1 + offset);
        int current_top_length = nb_in + 1 + offset; // len(<args>) + len(instance:method())=1 + len(instance)=0/1
        lua_pushvalue(L, -current_top_length);           // stack = [...,<args>, instance:method() (, instance), <args>]
        lua_remove(L, -(current_top_length+1));          // stack = [...,instance:method()(, instance), <args>]
    }
    CHECK_ERROR(lua_pcall(L, offset+nb_in, nb_out, 0));  // stack = [...,<returns>] (Function called)

    ASSERT_STATE(lua_gettop(L) == stack_size - nb_in + nb_out); // Sanity check
}


template<typename T>
void TorchVM::pop_lua_array(
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

// Instanciate templates explicitly here (avoid linker errors)
template char populate_number<char>(lua_State*);
template int populate_number<int>(lua_State*);
template long populate_number<long>(lua_State*);
template short populate_number<short>(lua_State*);
template float populate_number<float>(lua_State*);
template double populate_number<double>(lua_State*);

template void TorchVM::pop_lua_array<std::string>(std::vector<std::string>&, std::string (*populate_fct)(lua_State*));
template void TorchVM::pop_lua_array<char>(std::vector<char>&, char (*populate_fct)(lua_State*));
template void TorchVM::pop_lua_array<int>(std::vector<int>&, int (*populate_fct)(lua_State*));
template void TorchVM::pop_lua_array<long>(std::vector<long>&, long (*populate_fct)(lua_State*));
template void TorchVM::pop_lua_array<short>(std::vector<short>&, short (*populate_fct)(lua_State*));
template void TorchVM::pop_lua_array<float>(std::vector<float>&, float (*populate_fct)(lua_State*));
template void TorchVM::pop_lua_array<double>(std::vector<double>&, double (*populate_fct)(lua_State*));


////////////////////////// OpenCv/Tensor manipulation API //////////////////////////


// Define the generics here
#include "tensor_all.cpp"


////////////////////////// Low level Level API //////////////////////////


lua_State* TorchVM::getL()
{
    return L;
}

void TorchVM::gc()
{
    lua_gc(L, LUA_GCCOLLECT, 0);
}

void TorchVM::push_ref(int instance_ref)
{
    lua_rawgeti(L, LUA_REGISTRYINDEX, instance_ref);
}


LuaException::LuaException(const std::string& message) : _message("Lua error: " + message)
{
}

const char* LuaException::what() const noexcept
{
    return _message.c_str();
}


} // End of namespace
