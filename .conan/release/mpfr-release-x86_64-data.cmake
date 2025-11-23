########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(mpfr_COMPONENT_NAMES "")
if(DEFINED mpfr_FIND_DEPENDENCY_NAMES)
  list(APPEND mpfr_FIND_DEPENDENCY_NAMES gmp)
  list(REMOVE_DUPLICATES mpfr_FIND_DEPENDENCY_NAMES)
else()
  set(mpfr_FIND_DEPENDENCY_NAMES gmp)
endif()
set(gmp_FIND_MODE "NO_MODULE")

########### VARIABLES #######################################################################
#############################################################################################
set(mpfr_PACKAGE_FOLDER_RELEASE "C:/Users/ALGO_SERVER/.conan2/p/mpfrd5cad8619469b/p")
set(mpfr_BUILD_MODULES_PATHS_RELEASE )


set(mpfr_INCLUDE_DIRS_RELEASE "${mpfr_PACKAGE_FOLDER_RELEASE}/include")
set(mpfr_RES_DIRS_RELEASE )
set(mpfr_DEFINITIONS_RELEASE )
set(mpfr_SHARED_LINK_FLAGS_RELEASE )
set(mpfr_EXE_LINK_FLAGS_RELEASE )
set(mpfr_OBJECTS_RELEASE )
set(mpfr_COMPILE_DEFINITIONS_RELEASE )
set(mpfr_COMPILE_OPTIONS_C_RELEASE )
set(mpfr_COMPILE_OPTIONS_CXX_RELEASE )
set(mpfr_LIB_DIRS_RELEASE "${mpfr_PACKAGE_FOLDER_RELEASE}/lib")
set(mpfr_BIN_DIRS_RELEASE )
set(mpfr_LIBRARY_TYPE_RELEASE STATIC)
set(mpfr_IS_HOST_WINDOWS_RELEASE 1)
set(mpfr_LIBS_RELEASE mpfr)
set(mpfr_SYSTEM_LIBS_RELEASE )
set(mpfr_FRAMEWORK_DIRS_RELEASE )
set(mpfr_FRAMEWORKS_RELEASE )
set(mpfr_BUILD_DIRS_RELEASE )
set(mpfr_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(mpfr_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${mpfr_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${mpfr_COMPILE_OPTIONS_C_RELEASE}>")
set(mpfr_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${mpfr_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${mpfr_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${mpfr_EXE_LINK_FLAGS_RELEASE}>")


set(mpfr_COMPONENTS_RELEASE )