cmake_minimum_required(VERSION 3.16)
project(merlin VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MERLIN_NATIVE    "Optimize for this CPU (-march=native)" ON)
option(MERLIN_OPENMP    "Evaluate population members in parallel" ON)
option(MERLIN_SANITIZE  "Build with address + undefined-behaviour sanitizers" OFF)
option(MERLIN_TESTS     "Build unit tests" ON)

add_compile_options(-Wall -Wextra -Wshadow -Wno-unused-parameter)
if(MERLIN_NATIVE AND NOT MERLIN_SANITIZE)
  add_compile_options(-march=native)
endif()
if(MERLIN_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

# ---- core library: everything except main() ----
add_library(merlin_core STATIC
  src/core/rng.c
  src/core/tern.c
  src/core/data.c
  src/core/util.c
  src/core/ckpt.c
  src/es/es.c
  src/nn/charmlp.c
  src/nn/rwkv.c
)
target_include_directories(merlin_core PUBLIC src)
target_link_libraries(merlin_core PUBLIC m)

if(MERLIN_OPENMP)
  find_package(OpenMP)
  if(OpenMP_C_FOUND)
    target_link_libraries(merlin_core PUBLIC OpenMP::OpenMP_C)
    target_compile_definitions(merlin_core PUBLIC MERLIN_HAVE_OPENMP=1)
  endif()
endif()

# ---- programs ----
add_executable(merlin_train tools/merlin_train.c)
target_link_libraries(merlin_train PRIVATE merlin_core)

# ---- tests ----
if(MERLIN_TESTS)
  enable_testing()
  foreach(t rng tern data es)
    add_executable(test_${t} tests/test_${t}.c)
    target_link_libraries(test_${t} PRIVATE merlin_core)
    add_test(NAME ${t} COMMAND test_${t})
  endforeach()
endif()
