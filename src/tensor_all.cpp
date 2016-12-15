// WARNING: DO NOT PUT ANY INCLUDE STATEMENT HERE (WE ARE INSIDE LuaWrap NAMESPACE)


// Byte tensor
#define LUAW_TYPE uint8_t
#define LUAW_NAME Byte
#include "tensor_base.cpp"
#undef LUAW_TYPE
#undef LUAW_NAME


// Char tensor
#define LUAW_TYPE char
#define LUAW_NAME Char
#include "tensor_base.cpp"
#undef LUAW_TYPE
#undef LUAW_NAME


// Short tensor
#define LUAW_TYPE short
#define LUAW_NAME Short
#include "tensor_base.cpp"
#undef LUAW_TYPE
#undef LUAW_NAME


// Float tensor
#define LUAW_TYPE float
#define LUAW_NAME Float
#include "tensor_base.cpp"
#undef LUAW_TYPE
#undef LUAW_NAME


// Double tensor
#define LUAW_TYPE double
#define LUAW_NAME Double
#include "tensor_base.cpp"
#undef LUAW_TYPE
#undef LUAW_NAME


// Int tensor
#define LUAW_TYPE int
#define LUAW_NAME Int
#include "tensor_base.cpp"
#undef LUAW_TYPE
#undef LUAW_NAME


// Long tensor
#define LUAW_TYPE long
#define LUAW_NAME Long
#include "tensor_base.cpp"
#undef LUAW_TYPE
#undef LUAW_NAME
