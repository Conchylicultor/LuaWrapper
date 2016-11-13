// WARNING: DO NOT PUT ANY INCLUDE STATEMENT HERE (WE ARE INSIDE LuaWrap NAMESPACE)

// Ensure that the includer set type and name
#if !defined(LUAW_TYPE) || !defined(LUAW_NAME)
#    error "Includer has to set LUAW_TYPE and LUAW_NAME"
#endif


LUAW_Tensor* LUAW_THType(create_tensor)(int w, int h, int c, int batch_size)
{
    (void)batch_size;

    LUAW_Tensor* output = LUAW_THTensor(newWithSize3d)(c, // Channels
                                                       h, // Height
                                                       w); // Width

    return output;
}

void LUAW_THType(push_tensor)(lua_State* L, LUAW_Tensor* tensor)
{
    luaT_pushudata(L, (void*)tensor, "torch." LUAW_STRINGIFY(LUAW_NAME) "Tensor"); // Send the tensor to lua (Should delegate memory management)
}


void LUAW_THType(setdefaulttensortype)(lua_State* L)
{
    int stack_size = lua_gettop(L);

    // TODO: Should be done with call_lua_method (Error checking)
    lua_getglobal(L, "torch");
    lua_pushstring(L, "setdefaulttensortype");
    lua_gettable(L,-2);
    lua_remove(L,-2);
    lua_pushstring(L, "torch." LUAW_STRINGIFY(LUAW_NAME) "Tensor");
    lua_pcall(L,1,0,0);

    ASSERT_STATE(lua_gettop(L) == stack_size);
}
