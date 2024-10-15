#CMakelists for the C++ library 
#Copyright (C) 2024 Yaniv Karta

cmake_minimum_required(VERSION 3.16)

project(elfplusplus)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")    


#set include path 
include_directories(include)
add_library(elfplusplus src/elfplusplus.cpp src/platform_helper.cpp )    
#add linked library dependency



set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

set_target_properties(elfplusplus PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    )

#install library
install(TARGETS elfplusplus
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
    
    )   

#link target library
target_link_libraries(elfplusplus elf sqlite3 pthread ) 
target_link_options(elfplusplus PUBLIC -Wl,-soname,libelfplusplus.so.1) 

#link shared

#target_compile_definitions(elfplusplus PUBLIC SQLITE_ENABLE_API_ARMOR) 

#link .so

target_link_libraries(elfplusplus -lsqlite3 -lgnutls -lpthread -lelf -lsqlite3 -lpthread ) 


#build examples executables:

add_subdirectory(examples)

#add_executable(example elf_walk.cpp examples/elf_walk.cpp)

#link target library

