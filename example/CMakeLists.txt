add_executable(${EXAMPLENAME} example.cpp)
#生成可执行文件

target_link_libraries(${EXAMPLENAME} 
${OpenCV_LIBS} 
Ceres::ceres)

install(TARGETS ${EXAMPLENAME}  RUNTIME DESTINATION bin)
#？？？

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
#设置可执行文件输出目录

if(NOT ANDROID)
set(INSTALL_SRCS ${INSTALL_SRCS} 
example.cpp)
endif()

install(FILES ${INSTALL_SRCS}
        DESTINATION example
        COMPONENT dev)
