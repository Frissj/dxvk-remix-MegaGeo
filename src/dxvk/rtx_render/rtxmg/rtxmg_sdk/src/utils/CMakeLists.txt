set(lib rtxmg_utils)

file(GLOB sources "*.cpp" "../include/rtxmg/utils/*.h")

add_library(${lib} OBJECT ${sources})
target_include_directories(${lib} PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)
target_link_libraries(${lib} donut_engine)
set_target_properties(${lib} PROPERTIES FOLDER RTXMG)
