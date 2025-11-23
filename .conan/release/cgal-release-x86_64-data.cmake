########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(cgal_COMPONENT_NAMES "")
if(DEFINED cgal_FIND_DEPENDENCY_NAMES)
  list(APPEND cgal_FIND_DEPENDENCY_NAMES Boost Eigen3 mpfr gmp)
  list(REMOVE_DUPLICATES cgal_FIND_DEPENDENCY_NAMES)
else()
  set(cgal_FIND_DEPENDENCY_NAMES Boost Eigen3 mpfr gmp)
endif()
set(Boost_FIND_MODE "NO_MODULE")
set(Eigen3_FIND_MODE "NO_MODULE")
set(mpfr_FIND_MODE "NO_MODULE")
set(gmp_FIND_MODE "NO_MODULE")

########### VARIABLES #######################################################################
#############################################################################################
set(cgal_PACKAGE_FOLDER_RELEASE "C:/Users/ALGO_SERVER/.conan2/p/cgal85bc3f03f7f05/p")
set(cgal_BUILD_MODULES_PATHS_RELEASE "${cgal_PACKAGE_FOLDER_RELEASE}/lib/cmake/CGAL/conan-official-cgal-variables.cmake")


set(cgal_INCLUDE_DIRS_RELEASE "${cgal_PACKAGE_FOLDER_RELEASE}/include")
set(cgal_RES_DIRS_RELEASE )
set(cgal_DEFINITIONS_RELEASE )
set(cgal_SHARED_LINK_FLAGS_RELEASE )
set(cgal_EXE_LINK_FLAGS_RELEASE )
set(cgal_OBJECTS_RELEASE )
set(cgal_COMPILE_DEFINITIONS_RELEASE )
set(cgal_COMPILE_OPTIONS_C_RELEASE )
set(cgal_COMPILE_OPTIONS_CXX_RELEASE )
set(cgal_LIB_DIRS_RELEASE "${cgal_PACKAGE_FOLDER_RELEASE}/lib")
set(cgal_BIN_DIRS_RELEASE )
set(cgal_LIBRARY_TYPE_RELEASE UNKNOWN)
set(cgal_IS_HOST_WINDOWS_RELEASE 1)
set(cgal_LIBS_RELEASE )
set(cgal_SYSTEM_LIBS_RELEASE )
set(cgal_FRAMEWORK_DIRS_RELEASE )
set(cgal_FRAMEWORKS_RELEASE )
set(cgal_BUILD_DIRS_RELEASE "${cgal_PACKAGE_FOLDER_RELEASE}/lib/cmake/CGAL")
set(cgal_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(cgal_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${cgal_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${cgal_COMPILE_OPTIONS_C_RELEASE}>")
set(cgal_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${cgal_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${cgal_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${cgal_EXE_LINK_FLAGS_RELEASE}>")


set(cgal_COMPONENTS_RELEASE )