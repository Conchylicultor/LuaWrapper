// WARNING: DO NOT PUT ANY INCLUDE STATEMENT HERE (WE ARE INSIDE LuaWrap NAMESPACE)


// Float tensor
#define LUAW_TYPE float
#define LUAW_NAME THFloat
#include "tensor_base.cpp"
#undef LUAW_TYPE
#undef LUAW_NAME


// Byte tensor
#define LUAW_TYPE uint8_t
#define LUAW_NAME THByte
#include "tensor_base.cpp"
#undef LUAW_TYPE
#undef LUAW_NAME
