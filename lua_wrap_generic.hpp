#ifndef LUA_WRAP_GENERIC_HPP
#define LUA_WRAP_GENERIC_HPP


/** Macro used to generate code compatible with the Torch tensor lib.
  * Use metaprograming to generate C Template
  */

// Concatenate two macros (base operator, version with underscore)
#define LUAW_CONCAT_OP(A, B) A##B
#define LUAW_CONCAT(A, B) LUAW_CONCAT_OP(A, B)
#define LUAW_CONCAT_U(A, B) LUAW_CONCAT(A, LUAW_CONCAT(_, B))


/** Defines shortcuts THTypeTensor and THTypeStorage
  */
#define LUAW_TNAME LUAW_CONCAT(LUAW_NAME, Tensor)
#define LUAW_SNAME LUAW_CONCAT(LUAW_NAME, Storage)

#define LUAW_METHOD(NAME)  LUAW_CONCAT_U(LUAW_NAME, NAME)
#define LUAW_TMETHOD(NAME) LUAW_CONCAT_U(LUAW_TNAME, NAME)
#define LUAW_SMETHOD(NAME) LUAW_CONCAT_U(LUAW_SNAME, NAME)


#endif
