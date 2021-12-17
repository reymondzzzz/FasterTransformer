# TensorRT is available only for Linux 64
set(TENSORRT_ROOT "/usr/local/tensorrt")


macro(find_component _component _library)
    find_library(${_component}_LIBRARIES NAMES ${_library}
            HINTS "${TENSORRT_ROOT}/lib"
            )

    if(${_component}_LIBRARIES)
        set(_target "tensorrt::${_library}")
        set(${_component}_FOUND TRUE)
        add_library(${_target} UNKNOWN IMPORTED)
        set_property(TARGET ${_target}
                APPEND PROPERTY IMPORTED_LOCATION "${${_component}_LIBRARIES}")
    else()
        if(TENSORRT_FIND_REQUIRED)
            message(FATAL_ERROR "Could not find Tensorrt library: ${_component}")
        endif()
    endif()

    mark_as_advanced(
            ${_component}_LIBRARIES
    )
endmacro()


find_path(TENSORRT_INCLUDE_DIR "NvInfer.h" "${TENSORRT_ROOT}/include")


if(TENSORRT_INCLUDE_DIR)
    set(TENSORRT_FOUND "YES")
    set(TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
    add_library(tensorrt INTERFACE)
    target_include_directories(tensorrt INTERFACE ${TENSORRT_INCLUDE_DIR})
    set_target_properties(tensorrt
            PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR})

    foreach(_library nvinfer nvinfer_plugin nvonnxparser)
        string(TOUPPER "${_library}" _component)
        find_component(${_component} ${_library})
        if(${_component}_FOUND)
            list(APPEND TENSORRT_LIBRARIES ${${_component}_LIBRARIES})
            target_link_libraries(tensorrt INTERFACE "tensorrt::${_library}")
        else()
            set(TENSORRT_FOUND "NO")
        endif()
    endforeach()
else()
    set(TENSORRT_FOUND "NO")
    if(TENSORRT_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find TensorRT includes")
    endif()
endif()

mark_as_advanced(
        TENSORRT_LIBRARIES
        TENSORRT_INCLUDE_DIRS
        TENSORRT_FOUND
)
