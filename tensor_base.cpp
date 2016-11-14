// WARNING: DO NOT PUT ANY INCLUDE STATEMENT HERE (WE ARE INSIDE LuaWrap NAMESPACE)

// Ensure that the includer set type and name
#if !defined(LUAW_TYPE) || !defined(LUAW_NAME)
#    error "Includer has to set LUAW_TYPE and LUAW_NAME"
#endif


LUAW_Tensor* TorchVM::LUAW_THType_(create_tensor)(int w, int h, int c, int batch_size)
{
    (void)batch_size;

    LUAW_Tensor* output = LUAW_THTensor_(newWithSize3d)(c, // Channels
                                                        h, // Height
                                                        w); // Width

    return output;
}

void TorchVM::LUAW_THType_(push_tensor)(LUAW_Tensor* tensor)
{
    luaT_pushudata(L, (void*)tensor, LUAW_TENSOR_STR); // Send the tensor to lua (Should delegate memory management)
}


void TorchVM::LUAW_THType_(setdefaulttensortype)()
{
    int stack_size = lua_gettop(L);

    lua_pushstring(L, LUAW_TENSOR_STR);
    call_lua_method(torch, "setdefaulttensortype", 1, 0, false);

    ASSERT_STATE(lua_gettop(L) == stack_size);
}

void TorchVM::LUAW_THTensor_(print)(LUAW_Tensor* tensor)
{
    lua_getglobal(L,"print");
    luaT_pushudata(L, (void*) tensor, LUAW_TENSOR_STR);
    lua_pcall(L,1,0,0);
}
