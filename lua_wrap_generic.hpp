#ifndef LUA_WRAP_GENERIC_HPP
#define LUA_WRAP_GENERIC_HPP


/** Macro used to generate code compatible with the Torch tensor lib.
  * Use metaprograming to generate C Template
  */

// Concatenate two macros (base operator, version with underscore)
#define LUAW_CONCAT_OP(A, B) A##B
#define LUAW_CONCAT(A, B) LUAW_CONCAT_OP(A, B)
#define LUAW_CONCAT_U(A, B) LUAW_CONCAT(A, LUAW_CONCAT(_, B))

// Stringify macro (usefull for printing the tensor type for instance)
#define LUAW_STRINGIFY_OP(x) #x
#define LUAW_STRINGIFY(x) LUAW_STRINGIFY_OP(x)


// Defines shortcuts THTypeTensor and THTypeStorage
#define LUAW_Tensor LUAW_CONCAT(TH, LUAW_CONCAT(LUAW_NAME, Tensor))
#define LUAW_Storage LUAW_CONCAT(TH, LUAW_CONCAT(LUAW_NAME, Storage))

#define LUAW_THType(NAME)  LUAW_CONCAT(TH, LUAW_CONCAT_U(LUAW_NAME, NAME))
#define LUAW_THTensor(NAME) LUAW_CONCAT_U(LUAW_Tensor, NAME)
#define LUAW_THStorage(NAME) LUAW_CONCAT_U(LUAW_Storage, NAME)


#endif
