set(CONDA_DIR $ENV{CONDA_PREFIX})

if(NOT CONDA_DIR)
    message(ERROR "Can't find conda path from CONDA_PREFIX, please check the environment settings")
else()
    message("CONDA_DIR: ${CONDA_DIR}")
endif()

if(NOT CONDA_ENV_NAME OR ${CONDA_DIR} STREQUAL "base")
    message("No specific conda environment, switch to base environment")
    set(CONDA_ENV_DIR "${CONDA_DIR}")
    set(CONDA_INCLUDE_DIR "${CONDA_DIR}/include")
    set(CONDA_LIBRARY_DIR "${CONDA_DIR}/lib")
else()
    message("Select environment ${CONDA_ENV_NAME}")
    set(CONDA_ENV_DIR "${CONDA_DIR}/env/${CONDA_ENV_NAME}")
    set(CONDA_INCLUDE_DIR "${CONDA_DIR}/env/${CONDA_ENV_NAME}/include")
    set(CONDA_LIBRARY_DIR "${CONDA_DIR}/env/${CONDA_ENV_NAME}/lib")
endif()

message("CONDA_ENV_DIR: ${CONDA_DIR}")
message("CONDA_INCLUDE_DIR: ${CONDA_INCLUDE_DIR}")
message("CONDA_LIBRARY_DIR: ${CONDA_LIBRARY_DIR}")

#find python version
FILE(GLOB python_file "${CONDA_INCLUDE_DIR}/python*")
string(REGEX MATCH "python([0-9]+).([0-9]+)" CONDA_PYTHON ${python_file})
message("Find python in conda: ${CONDA_PYTHON}")

set(PYTHON_INCLUDE_DIR "${CONDA_INCLUDE_DIR}/${CONDA_PYTHON}")
set(PYTHON_LIB "${CONDA_LIBRARY_DIR}/lib${CONDA_PYTHON}.so")

message("PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}")
message("PYTHON_LIB: ${PYTHON_LIB}")