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
LUAW_Tensor* LUAW_THType(create_tensor)(int w, int h, int c=3, int batch_size=1);

/** Push a Tensor on top of the stack
  */
void LUAW_THType(push_tensor)(lua_State* L, LUAW_Tensor* tensor);
