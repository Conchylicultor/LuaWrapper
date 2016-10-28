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



/** Macro which assert if the condition is valid. Mainly called to check if the lua stack
  * is in a valid state
  */
#define ASSERT_STATE(valid) \
    if (!(valid)) \
    {\
        throw LuaException("Invalid state in " + std::string(__FILE__) + " at line " + std::to_string(__LINE__) + ": " LUAW_STRINGIFY(valid));\
    }


// TODO: Could create class instead ?
namespace LuaWrap
{
    ////////////////////////// High Level API //////////////////////////

    /** Initialize lua state, and load standard libs
      * Will load torch and nn and set default tensor type to FloatTensor
      */
    lua_State* init_torch_vm();

    /** Equivalent to require 'script_name'.
      * Return the lua register id of the script
      * The script should follow some constraint: It should return an object
      * containing 2 methods:
      *  - load(): the init method (will be called with this function)
      *  - forward(batch_in): the image to process
      * Return the lua register id of the model
      * Warning: the path is given from the working directory
      */
    int load_script(lua_State* L, const std::string& script_name);

    /** Equivallent to torch.load('model_name')
      * Warning: the path is given from the working directory
      */
    int load_model(lua_State* L, const std::string& model_name);

    /** Equivalent to require 'lib_name'
      */
    void load_lualib(lua_State* L, const std::string& lib_name);

    /**
      * @brief call_lua_method A wrapper around lua for easy method calling [-nb_in, +nb_out, -]
      * WARNING: If the number of arguments is too important, and the stack is already filled, it could overflow
      * @param L             lua state
      * @param instance_ref  a reference on the object for which calling the method
      * @param method_name   the name of the method to call
      * @param nb_in         nb of arguments for the method (WARNING: Those have to be pushed on top of
      *                      the stack before calling this function). Do not include the 'self' inplicit argument
      * @param nb_out        Nb of returns values (you can catch them at the top of the stack)
      * @param is_class      True is the the table act as a class (will add self argument), false if it's just a regular table
      */
    void call_lua_method(
        lua_State* L,
        int instance_ref,
        const std::string& method_name,
        int nb_in=0,
        int nb_out=0,
        bool is_class=true
    );

    /** Extract the array at the top of the stack [-1, +0, -].
      * The values are added to out_array. out_array is cleared before being feeded
      */
    template<typename T>
    void pop_lua_array(
        lua_State* L,
        std::vector<T>& out_array,
        T (*populate_fct)(lua_State*) // Read the value on top of the stack and return it [-0,+0,-]
    );


    /** Helper class for pop_lua_array if array contains string
      */
    std::string populate_string(lua_State* L);


    /** Helper class for pop_lua_array if array contains numbers (float, int)
      */
    template <typename TNumber>
    TNumber populate_number(lua_State* L);

    ////////////////////////// OpenCv/Tensor manipulation API //////////////////////////

    // Define the generics here
    #include "tensor_all.hpp"

    ////////////////////////// Low level Level API //////////////////////////

    /** If returnedValue != 0, try to capture the error from lua.
      * Raise an exception containing the error message
      * WARNING: If the exception is thrown, the stack will be left in an unknown state
      */
    void check_error(int returnedValue, lua_State *L);

    /** For debugging purpose
      */
    void print_tensor(lua_State* L, THFloatTensor* tensor);

    /**
      */
    void set_defaultfloattensor(lua_State* L);

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
