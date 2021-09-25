cmake_minimum_required(VERSION 3.15...3.20)

project(
  BenchLinearSolver
  VERSION "0.0.1"
  DESCRIPTION "Benchmark of lib/methods for solving linear system."
  LANGUAGES C CXX)
  
include(FetchContent) 
include(CheckCXXCompilerFlag)
include(CMakePrintHelpers)


# this variable controls the default value of the options which are normally set
# to ON (useful if you want to configure a minimal version of Gmsh: e.g. "cmake
# -DDEFAULT=0 -DENABLE_POST=1 -DENABLE_PARSER=1")
set(DEFAULT ON CACHE INTERNAL "Default value for enabled-by-default options")



################################## MACRO AND FUNCTIONS ####################################

macro(opt OPTION HELP VALUE)
  option(ENABLE_${OPTION} ${HELP} ${VALUE})
  set(OPT_TEXI "${OPT_TEXI}\n@item ENABLE_${OPTION}\n${HELP} (default: ${VALUE})")
  message("Compilation option : " ${OPT_TEXI} "is set to : " ${VALUE})
endmacro()


function(enable_cxx_compiler_flag_if_supported flag)
  string(FIND "${CMAKE_CXX_FLAGS}" "${flag}" flag_already_set)
  if(flag_already_set EQUAL -1)
    check_cxx_compiler_flag("${flag}" flag_supported)
    if(flag_supported)
      set(CMAKE_CXX_FLAGS
          "${CMAKE_CXX_FLAGS} ${flag}"
          PARENT_SCOPE)
    endif()
    unset(flag_supported CACHE)
  endif()
endfunction()


# Only do these if this is the main project, and not if it is included through
# add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  # Optionally set things like CMAKE_CXX_STANDARD,
  # CMAKE_POSITION_INDEPENDENT_CODE here

  # Let's ensure -std=c++xx instead of -std=g++xx
  set(CMAKE_CXX_EXTENSIONS OFF)

  # Let's nicely support folders in IDEs
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  ### Handling build type nicely
  # Set a default build type if none was specified
  set(default_build_type "Release")
  if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
    set(default_build_type "RelWithDebInfo")
  endif()

  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
        STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
  endif()

  if(ENABLE_CLANG_TIDY)
    message("Clang tidy at compilation is enabled")
    set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=*,clang-analyzer*,-clang-analyzer-cplusplus*)
  endif()


  # Compiler tags
  enable_cxx_compiler_flag_if_supported("-Wall")
  enable_cxx_compiler_flag_if_supported("-Wextra")
  enable_cxx_compiler_flag_if_supported("-pedantic")
  enable_cxx_compiler_flag_if_supported("-Wshadow")
  enable_cxx_compiler_flag_if_supported("-Wnon-virtual-dtor")
  enable_cxx_compiler_flag_if_supported("-Wnon-virtual-dtor")
  enable_cxx_compiler_flag_if_supported("-Wno-deprecated-copy")
  enable_cxx_compiler_flag_if_supported("-fsanitize=address")
  

endif()

# Find OpenMP
if(ENABLE_OPEN_MP)
  message("OpenMP optimization activated.")
  find_package(OpenMP)
  if(OpenMP_FOUND)
    set(USE_OPENMP_ACCELERATION 1)
  endif()
else()
  set(USE_OPENMP_ACCELERATION 0)
endif(ENABLE_OPEN_MP)

# Find or get benchark lib
find_package(benchmark)
if(NOT benchmark_FOUND)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
    message("Fetching google benchmark lib ...")
    FetchContent_MakeAvailable(benchmark) 
endif()
  
#  Find Eigen library
find_package(Eigen3 REQUIRED NO_MODULE)

add_subdirectory(benchmarks)

