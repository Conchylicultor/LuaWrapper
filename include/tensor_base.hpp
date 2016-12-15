// No include guards (Done in tensor_all.hpp)

// WARNING: DO NOT PUT ANY INCLUDE STATEMENT HERE (WE ARE INSIDE LuaWrap NAMESPACE)

// Ensure that the includer set type and name
#if !defined(LUAW_TYPE) || !defined(LUAW_NAME)
#    error "Includer has to set LUAW_TYPE and LUAW_NAME"
#endif

/** Return a Torch Tensor
  * Memory should be freed by lua garbadge collector
  * WARNING: Batch size not supported
  */
LUAW_Tensor* LUAW_THType_(create_tensor3d)(int c, int w, int h); // TODO: Should not be a member of torch VM (just in the namespace) < Easy fix: static member

/** Define the default tensor type
  */
void LUAW_THType_(setdefaulttensortype)();

/** Push a Tensor on top of the stack
  */
void LUAW_THTensor_(push)(LUAW_Tensor* tensor);

/** Pop a Tensor and return it
  */
LUAW_Tensor* LUAW_THTensor_(pop)();

/** Print a tensor using the Lua interface.
  */
void LUAW_THTensor_(print)(LUAW_Tensor* tensor);
