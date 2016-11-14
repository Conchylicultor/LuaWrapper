#ifndef LUA_WRAP_HPP
#define LUA_WRAP_HPP

#include <string>
#include <vector>
#include <exception>

#include <lua.hpp>
extern "C" {
    #include <TH.h>
    #include <luaT.h>
}

#include "lua_wrap_generic.hpp"



// TODO: can add an option to call luaGC every x iters of callLuaMethod

namespace LuaWrap
{

class TorchVM
{
public:
    ////////////////////////// High Level API //////////////////////////

    /** Initialize lua state, and load standard libs
      * Will load torch and nn and set default tensor type to FloatTensor
      */
    TorchVM();

    /** Close the lua state
      */
    ~TorchVM();

    /** Equivalent to require 'script_name'.
      * Return the lua register id of the script
      * The script should follow some constraint: It should return an object
      * containing 2 methods:
      *  - load(): the init method (will be called with this function)
      *  - forward(batch_in): the image to process
      * Return the lua register id of the model
      * Warning: the path is given from the working directory
      */
    int load_script(const std::string& script_name);

    /** Equivallent to torch.load('model_name')
      * Warning: the path is given from the working directory
      */
    int load_model(const std::string& model_name);

    /** Equivalent to require 'lib_name'
      */
    void load_lualib(const std::string& lib_name);

    /**
      * @brief call_lua_method A wrapper around lua for easy method calling [-nb_in, +nb_out, -]
      * WARNING: If the number of arguments is too important, and the stack is already filled, it could overflow
      * @param instance_ref  a reference on the object for which calling the method (LUA_NOREF if calling from global scope)
      * @param method_name   the name of the method to call
      * @param nb_in         nb of arguments for the method (WARNING: Those have to be pushed on top of
      *                      the stack before calling this function). Do not include the 'self' inplicit argument
      * @param nb_out        Nb of returns values (you can catch them at the top of the stack)
      * @param is_class      True is the the table act as a class (will add self argument), false if it's just a regular table. In lua that would be a:foo() vs a.foo()
      */
    void call_lua_method(
        int instance_ref,
        const std::string& method_name,
        int nb_in=0,
        int nb_out=0,
        bool is_class=false
    );

    /** Extract the array at the top of the stack [-1, +0, -].
      * The values are added to out_array. out_array is cleared before being feeded
      */
    template<typename T>
    void pop_lua_array(
        std::vector<T>& out_array,
        T (*populate_fct)(lua_State*) // Read the value on top of the stack and return it [-0,+0,-]
    );


    ////////////////////////// OpenCv/Tensor manipulation API //////////////////////////

    // Define the generics here
    #include "tensor_all.hpp"

    ////////////////////////// Low level Level API //////////////////////////

    /** For more control over the stack
      */
    lua_State* getL();

    /** Garbadge collect
      */
    void gc();

private:
    lua_State* L;
    int torch;  // Reference on the torch lib

    int counter_L;  // Manage the lifecycle for L TODO: Copy constructor
};


/** Helper class for pop_lua_array if array contains string
  */
std::string populate_string(lua_State* L);


/** Helper class for pop_lua_array if array contains numbers (float, int)
  */
template <typename TNumber>
TNumber populate_number(lua_State* L);


/** Exception raised when an error is detected lua
  */
class LuaException: public std::exception
{
public:
    LuaException(const std::string& message);

    virtual const char* what() const noexcept;

private:
    std::string _message;
};

}


#endif
