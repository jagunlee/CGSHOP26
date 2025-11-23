########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

list(APPEND gmp_COMPONENT_NAMES gmp::libgmp gmp::gmpxx)
list(REMOVE_DUPLICATES gmp_COMPONENT_NAMES)
if(DEFINED gmp_FIND_DEPENDENCY_NAMES)
  list(APPEND gmp_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES gmp_FIND_DEPENDENCY_NAMES)
else()
  set(gmp_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(gmp_PACKAGE_FOLDER_RELEASE "C:/Users/ALGO_SERVER/.conan2/p/gmp772bf963b9314/p")
set(gmp_BUILD_MODULES_PATHS_RELEASE )


set(gmp_INCLUDE_DIRS_RELEASE "${gmp_PACKAGE_FOLDER_RELEASE}/include")
set(gmp_RES_DIRS_RELEASE )
set(gmp_DEFINITIONS_RELEASE )
set(gmp_SHARED_LINK_FLAGS_RELEASE )
set(gmp_EXE_LINK_FLAGS_RELEASE )
set(gmp_OBJECTS_RELEASE )
set(gmp_COMPILE_DEFINITIONS_RELEASE )
set(gmp_COMPILE_OPTIONS_C_RELEASE )
set(gmp_COMPILE_OPTIONS_CXX_RELEASE )
set(gmp_LIB_DIRS_RELEASE "${gmp_PACKAGE_FOLDER_RELEASE}/lib")
set(gmp_BIN_DIRS_RELEASE )
set(gmp_LIBRARY_TYPE_RELEASE STATIC)
set(gmp_IS_HOST_WINDOWS_RELEASE 1)
set(gmp_LIBS_RELEASE gmpxx gmp)
set(gmp_SYSTEM_LIBS_RELEASE )
set(gmp_FRAMEWORK_DIRS_RELEASE )
set(gmp_FRAMEWORKS_RELEASE )
set(gmp_BUILD_DIRS_RELEASE )
set(gmp_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(gmp_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${gmp_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${gmp_COMPILE_OPTIONS_C_RELEASE}>")
set(gmp_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${gmp_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${gmp_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${gmp_EXE_LINK_FLAGS_RELEASE}>")


set(gmp_COMPONENTS_RELEASE gmp::libgmp gmp::gmpxx)
########### COMPONENT gmp::gmpxx VARIABLES ############################################

set(gmp_gmp_gmpxx_INCLUDE_DIRS_RELEASE "${gmp_PACKAGE_FOLDER_RELEASE}/include")
set(gmp_gmp_gmpxx_LIB_DIRS_RELEASE "${gmp_PACKAGE_FOLDER_RELEASE}/lib")
set(gmp_gmp_gmpxx_BIN_DIRS_RELEASE )
set(gmp_gmp_gmpxx_LIBRARY_TYPE_RELEASE STATIC)
set(gmp_gmp_gmpxx_IS_HOST_WINDOWS_RELEASE 1)
set(gmp_gmp_gmpxx_RES_DIRS_RELEASE )
set(gmp_gmp_gmpxx_DEFINITIONS_RELEASE )
set(gmp_gmp_gmpxx_OBJECTS_RELEASE )
set(gmp_gmp_gmpxx_COMPILE_DEFINITIONS_RELEASE )
set(gmp_gmp_gmpxx_COMPILE_OPTIONS_C_RELEASE "")
set(gmp_gmp_gmpxx_COMPILE_OPTIONS_CXX_RELEASE "")
set(gmp_gmp_gmpxx_LIBS_RELEASE gmpxx)
set(gmp_gmp_gmpxx_SYSTEM_LIBS_RELEASE )
set(gmp_gmp_gmpxx_FRAMEWORK_DIRS_RELEASE )
set(gmp_gmp_gmpxx_FRAMEWORKS_RELEASE )
set(gmp_gmp_gmpxx_DEPENDENCIES_RELEASE gmp::libgmp)
set(gmp_gmp_gmpxx_SHARED_LINK_FLAGS_RELEASE )
set(gmp_gmp_gmpxx_EXE_LINK_FLAGS_RELEASE )
set(gmp_gmp_gmpxx_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(gmp_gmp_gmpxx_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${gmp_gmp_gmpxx_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${gmp_gmp_gmpxx_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${gmp_gmp_gmpxx_EXE_LINK_FLAGS_RELEASE}>
)
set(gmp_gmp_gmpxx_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${gmp_gmp_gmpxx_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${gmp_gmp_gmpxx_COMPILE_OPTIONS_C_RELEASE}>")
########### COMPONENT gmp::libgmp VARIABLES ############################################

set(gmp_gmp_libgmp_INCLUDE_DIRS_RELEASE "${gmp_PACKAGE_FOLDER_RELEASE}/include")
set(gmp_gmp_libgmp_LIB_DIRS_RELEASE "${gmp_PACKAGE_FOLDER_RELEASE}/lib")
set(gmp_gmp_libgmp_BIN_DIRS_RELEASE )
set(gmp_gmp_libgmp_LIBRARY_TYPE_RELEASE STATIC)
set(gmp_gmp_libgmp_IS_HOST_WINDOWS_RELEASE 1)
set(gmp_gmp_libgmp_RES_DIRS_RELEASE )
set(gmp_gmp_libgmp_DEFINITIONS_RELEASE )
set(gmp_gmp_libgmp_OBJECTS_RELEASE )
set(gmp_gmp_libgmp_COMPILE_DEFINITIONS_RELEASE )
set(gmp_gmp_libgmp_COMPILE_OPTIONS_C_RELEASE "")
set(gmp_gmp_libgmp_COMPILE_OPTIONS_CXX_RELEASE "")
set(gmp_gmp_libgmp_LIBS_RELEASE gmp)
set(gmp_gmp_libgmp_SYSTEM_LIBS_RELEASE )
set(gmp_gmp_libgmp_FRAMEWORK_DIRS_RELEASE )
set(gmp_gmp_libgmp_FRAMEWORKS_RELEASE )
set(gmp_gmp_libgmp_DEPENDENCIES_RELEASE )
set(gmp_gmp_libgmp_SHARED_LINK_FLAGS_RELEASE )
set(gmp_gmp_libgmp_EXE_LINK_FLAGS_RELEASE )
set(gmp_gmp_libgmp_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(gmp_gmp_libgmp_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${gmp_gmp_libgmp_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${gmp_gmp_libgmp_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${gmp_gmp_libgmp_EXE_LINK_FLAGS_RELEASE}>
)
set(gmp_gmp_libgmp_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${gmp_gmp_libgmp_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${gmp_gmp_libgmp_COMPILE_OPTIONS_C_RELEASE}>")