set(lib subdivision)
set(folder RTXMG)

file(GLOB sources "*.cpp" "../include/rtxmg/${lib}/*.h" "../include/rtxmg/${lib}/osd_ports/tmr/*.h" "../include/rtxmg/${lib}/*.hlsli" *.cfg)

add_library(${lib} OBJECT ${sources})
target_include_directories(${lib} PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)
target_link_libraries(${lib} donut_engine profiler osd_lite_static implot)
set_target_properties(${lib} PROPERTIES FOLDER ${folder})