project(tests)
message("-- Project added: ${PROJECT_NAME}")

ADD_EXECUTABLE(cpu_spmm spmm.cc)
target_link_libraries(cpu_spmm dmlc -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)
