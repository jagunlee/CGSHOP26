# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(cgal_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(cgal_FRAMEWORKS_FOUND_RELEASE "${cgal_FRAMEWORKS_RELEASE}" "${cgal_FRAMEWORK_DIRS_RELEASE}")

set(cgal_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET cgal_DEPS_TARGET)
    add_library(cgal_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET cgal_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${cgal_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${cgal_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:boost::boost;Eigen3::Eigen;mpfr::mpfr;gmp::gmp>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### cgal_DEPS_TARGET to all of them
conan_package_library_targets("${cgal_LIBS_RELEASE}"    # libraries
                              "${cgal_LIB_DIRS_RELEASE}" # package_libdir
                              "${cgal_BIN_DIRS_RELEASE}" # package_bindir
                              "${cgal_LIBRARY_TYPE_RELEASE}"
                              "${cgal_IS_HOST_WINDOWS_RELEASE}"
                              cgal_DEPS_TARGET
                              cgal_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "cgal"    # package_name
                              "${cgal_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${cgal_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## GLOBAL TARGET PROPERTIES Release ########################################
    set_property(TARGET CGAL::CGAL
                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Release>:${cgal_OBJECTS_RELEASE}>
                 $<$<CONFIG:Release>:${cgal_LIBRARIES_TARGETS}>
                 )

    if("${cgal_LIBS_RELEASE}" STREQUAL "")
        # If the package is not declaring any "cpp_info.libs" the package deps, system libs,
        # frameworks etc are not linked to the imported targets and we need to do it to the
        # global target
        set_property(TARGET CGAL::CGAL
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     cgal_DEPS_TARGET)
    endif()

    set_property(TARGET CGAL::CGAL
                 APPEND PROPERTY INTERFACE_LINK_OPTIONS
                 $<$<CONFIG:Release>:${cgal_LINKER_FLAGS_RELEASE}>)
    set_property(TARGET CGAL::CGAL
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                 $<$<CONFIG:Release>:${cgal_INCLUDE_DIRS_RELEASE}>)
    # Necessary to find LINK shared libraries in Linux
    set_property(TARGET CGAL::CGAL
                 APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                 $<$<CONFIG:Release>:${cgal_LIB_DIRS_RELEASE}>)
    set_property(TARGET CGAL::CGAL
                 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                 $<$<CONFIG:Release>:${cgal_COMPILE_DEFINITIONS_RELEASE}>)
    set_property(TARGET CGAL::CGAL
                 APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                 $<$<CONFIG:Release>:${cgal_COMPILE_OPTIONS_RELEASE}>)

########## For the modules (FindXXX)
set(cgal_LIBRARIES_RELEASE CGAL::CGAL)
