#ifndef LUA_WRAP_GENERIC_HPP
#define LUA_WRAP_GENERIC_HPP


/** Macro used to generate code compatible with the Torch tensor lib.
  * Use metaprograming to generate C Template
  */

// Concatenate two macros (base operator, version with underscore)
#define LUAW_CONCAT_OP(A, B) A##B
#define LUAW_CONCAT(A, B) LUAW_CONCAT_OP(A, B)
#define LUAW_CONCAT_U(A, B) LUAW_CONCAT(A, LUAW_CONCAT(_, B))

// Stringify macro (usefull for printing the tensor type)
#define LUAW_STRINGIFY_OP(x) #x
#define LUAW_STRINGIFY(x) LUAW_STRINGIFY_OP(x)


// Defines shortcuts THTypeTensor and THTypeStorage
#define LUAW_THNAME LUAW_CONCAT(TH, LUAW_NAME)
#define LUAW_Tensor LUAW_CONCAT(LUAW_THNAME, Tensor)
#define LUAW_Storage LUAW_CONCAT(LUAW_THNAME, Storage)

#define LUAW_THType_(NAME) LUAW_CONCAT_U(LUAW_THNAME, NAME)
#define LUAW_THTensor_(NAME) LUAW_CONCAT_U(LUAW_Tensor, NAME)
#define LUAW_THStorage_(NAME) LUAW_CONCAT_U(LUAW_Storage, NAME)

// Tensor name
#define LUAW_TENSOR_STR ("torch." LUAW_STRINGIFY(LUAW_NAME) "Tensor")

#endif
