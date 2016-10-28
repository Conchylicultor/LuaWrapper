#ifndef TENSOR_ALL_HPP
#define TENSOR_ALL_HPP

// WARNING: DO NOT PUT ANY INCLUDE STATEMENT HERE (WE ARE INSIDE LuaWrap NAMESPACE)


// Float tensor
#define LUAW_TYPE float
#define LUAW_NAME Float
#include "tensor_base.hpp"
#undef LUAW_TYPE
#undef LUAW_NAME


// Byte tensor
#define LUAW_TYPE uint8_t
#define LUAW_NAME Byte
#include "tensor_base.hpp"
#undef LUAW_TYPE
#undef LUAW_NAME


#endif
