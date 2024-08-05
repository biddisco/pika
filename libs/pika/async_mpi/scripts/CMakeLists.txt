# this file needs to have paths updated with correct build dir etc
configure_file(benchmark_mpi_async.py ${PROJECT_BINARY_DIR}/scripts/benchmark_mpi_async.py @ONLY)

set(mpix_helper_files systems.py miniapps.py gh200-gpu-wrapper.sh)
foreach(file ${mpix_helper_files})
  configure_file(${file} ${PROJECT_BINARY_DIR}/scripts/${file} COPYONLY)
endforeach()
