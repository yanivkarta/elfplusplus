#cmake for examples : 

cmake_minimum_required(VERSION 3.16)

project(elfplusplus)


#set source folder
#add_subdirectory(examples)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")


#set include path
include_directories(../include)


add_executable(elf_walk elf_walk.cpp ../examples/elf_walk.cpp)

#add executable with library dependency

add_executable(feature_extractor elf_feature_extractor.cpp ../examples/elf_feature_extractor.cpp) 
target_link_libraries(feature_extractor elfplusplus sqlite3 gnutls pthread) 
add_executable(elfmonitor elfmonitor.cpp ../examples/elfmonitor.cpp) 

target_link_libraries(elfmonitor elfplusplus sqlite3 gnutls pthread) 


add_executable(elfdump elfdump.cpp ../examples/elfdump.cpp)
target_link_libraries(elfdump elfplusplus sqlite3 gnutls pthread )


