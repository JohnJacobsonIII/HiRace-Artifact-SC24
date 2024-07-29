set(hirace_SOURCES
  CudaRWMain.cpp
  CudaRW.cpp
)

add_clang_executable(hirace
  ${hirace_SOURCES}
)

target_link_libraries(hirace
  PRIVATE
  clangTooling
)

target_include_directories(hirace
  PRIVATE
  include
)
