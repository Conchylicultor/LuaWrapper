// WARNING: DO NOT PUT ANY INCLUDE STATEMENT HERE (WE ARE INSIDE LuaWrap NAMESPACE)

// Ensure that the includer set type and name
#if !defined(LUAW_TYPE) || !defined(LUAW_NAME)
#    error "Includer has to set LUAW_TYPE and LUAW_NAME"
#endif


LUAW_TNAME* LUAW_METHOD(create_tensor)(int w, int h, int c, int batch_size)
{
    (void)batch_size;

    int len = h * w * c;
    long stride_1 = h * w;
    long stride_2 = w;
    long stride_3 = 1;

    //LUAW_TYPE* tensorData = (LUAW_TYPE*)malloc(sizeof(LUAW_TYPE)*len);
    LUAW_SNAME* storage = LUAW_SMETHOD(newWithSize)(len);
    LUAW_TNAME* input = LUAW_TMETHOD(newWithStorage3d)(storage,
                                                       0, // Offset
                                                       c, stride_1, // Channels
                                                       h, stride_2, // Height
                                                       w, stride_3); // Width

    return input;
}

void LUAW_METHOD(push_tensor)(lua_State* L, LUAW_TNAME* tensor)
{
    luaT_pushudata(L, (void*)tensor, "torch." LUAW_STRINGIFY(LUAW_NAME) "Tensor"); // Send the tensor to lua (Should delegate memory management)
}
