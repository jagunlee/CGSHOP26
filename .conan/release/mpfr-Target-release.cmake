# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(mpfr_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(mpfr_FRAMEWORKS_FOUND_RELEASE "${mpfr_FRAMEWORKS_RELEASE}" "${mpfr_FRAMEWORK_DIRS_RELEASE}")

set(mpfr_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET mpfr_DEPS_TARGET)
    add_library(mpfr_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET mpfr_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${mpfr_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${mpfr_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:gmp::gmp>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### mpfr_DEPS_TARGET to all of them
conan_package_library_targets("${mpfr_LIBS_RELEASE}"    # libraries
                              "${mpfr_LIB_DIRS_RELEASE}" # package_libdir
                              "${mpfr_BIN_DIRS_RELEASE}" # package_bindir
                              "${mpfr_LIBRARY_TYPE_RELEASE}"
                              "${mpfr_IS_HOST_WINDOWS_RELEASE}"
                              mpfr_DEPS_TARGET
                              mpfr_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "mpfr"    # package_name
                              "${mpfr_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${mpfr_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## GLOBAL TARGET PROPERTIES Release ########################################
    set_property(TARGET mpfr::mpfr
                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Release>:${mpfr_OBJECTS_RELEASE}>
                 $<$<CONFIG:Release>:${mpfr_LIBRARIES_TARGETS}>
                 )

    if("${mpfr_LIBS_RELEASE}" STREQUAL "")
        # If the package is not declaring any "cpp_info.libs" the package deps, system libs,
        # frameworks etc are not linked to the imported targets and we need to do it to the
        # global target
        set_property(TARGET mpfr::mpfr
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     mpfr_DEPS_TARGET)
    endif()

    set_property(TARGET mpfr::mpfr
                 APPEND PROPERTY INTERFACE_LINK_OPTIONS
                 $<$<CONFIG:Release>:${mpfr_LINKER_FLAGS_RELEASE}>)
    set_property(TARGET mpfr::mpfr
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                 $<$<CONFIG:Release>:${mpfr_INCLUDE_DIRS_RELEASE}>)
    # Necessary to find LINK shared libraries in Linux
    set_property(TARGET mpfr::mpfr
                 APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                 $<$<CONFIG:Release>:${mpfr_LIB_DIRS_RELEASE}>)
    set_property(TARGET mpfr::mpfr
                 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                 $<$<CONFIG:Release>:${mpfr_COMPILE_DEFINITIONS_RELEASE}>)
    set_property(TARGET mpfr::mpfr
                 APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                 $<$<CONFIG:Release>:${mpfr_COMPILE_OPTIONS_RELEASE}>)

########## For the modules (FindXXX)
set(mpfr_LIBRARIES_RELEASE mpfr::mpfr)
