// WARNING: DO NOT PUT ANY INCLUDE STATEMENT HERE (WE ARE INSIDE LuaWrap NAMESPACE)

// Ensure that the includer set type and name
#if !defined(LUAW_TYPE) || !defined(LUAW_NAME)
#    error "Includer has to set LUAW_TYPE and LUAW_NAME"
#endif


LUAW_Tensor* TorchVM::LUAW_THType_(create_tensor3d)(int c, int w, int h)
{
    // The function can seem useless but ensure that the tensor is created with the right
    // method. Otherwise creating an intermediate storage first will lead to a memory leak
    LUAW_Tensor* output = LUAW_THTensor_(newWithSize3d)(c, // Channels
                                                        h, // Height
                                                        w); // Width

    return output;
}

void TorchVM::LUAW_THType_(setdefaulttensortype)()
{
    int stack_size = lua_gettop(L);

    lua_pushstring(L, LUAW_TENSOR_STR);
    call_lua_method(torch, "setdefaulttensortype", 1, 0, false);

    ASSERT_STATE(lua_gettop(L) == stack_size);
}

void TorchVM::LUAW_THTensor_(push)(LUAW_Tensor* tensor)
{
    luaT_pushudata(L, (void*)tensor, LUAW_TENSOR_STR); // Send the tensor to lua (Should delegate memory management)
}

LUAW_Tensor* TorchVM::LUAW_THTensor_(pop)()
{
    LUAW_Tensor* tensor = (LUAW_Tensor*) luaT_checkudata(L, 1, LUAW_TENSOR_STR);
    lua_pop(L,1); //luaT_checkudata doesnt actually pop the value
    // TODO: Does pop trigger the garbadge collector ?? (if yes, then we are in trouble)
    return tensor;
}

void TorchVM::LUAW_THTensor_(print)(LUAW_Tensor* tensor)
{
    int stack_size = lua_gettop(L);

    luaT_pushudata(L, (void*) tensor, LUAW_TENSOR_STR);
    call_lua_method(LUA_NOREF, "print", 1, 0);

    ASSERT_STATE(lua_gettop(L) == stack_size);
}
