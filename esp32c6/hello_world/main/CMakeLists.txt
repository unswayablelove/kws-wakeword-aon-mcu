idf_component_register(SRCS "hello_world_main.c" "kws/kws_e2e_nofpu_v3.c" "kws/kws_decide.c"
                       INCLUDE_DIRS "" "kws"
                       PRIV_REQUIRES esp_driver_i2s esp_timer)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error=misleading-indentation -Wno-misleading-indentation)
