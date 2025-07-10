#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "mcl::mcl" for configuration "Release"
set_property(TARGET mcl::mcl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(mcl::mcl PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmcl.1.73.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libmcl.1.dylib"
  )

list(APPEND _cmake_import_check_targets mcl::mcl )
list(APPEND _cmake_import_check_files_for_mcl::mcl "${_IMPORT_PREFIX}/lib/libmcl.1.73.dylib" )

# Import target "mcl::mcl_st" for configuration "Release"
set_property(TARGET mcl::mcl_st APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(mcl::mcl_st PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmcl.a"
  )

list(APPEND _cmake_import_check_targets mcl::mcl_st )
list(APPEND _cmake_import_check_files_for_mcl::mcl_st "${_IMPORT_PREFIX}/lib/libmcl.a" )

# Import target "mcl::mclbn256" for configuration "Release"
set_property(TARGET mcl::mclbn256 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(mcl::mclbn256 PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmclbn256.1.73.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libmclbn256.1.dylib"
  )

list(APPEND _cmake_import_check_targets mcl::mclbn256 )
list(APPEND _cmake_import_check_files_for_mcl::mclbn256 "${_IMPORT_PREFIX}/lib/libmclbn256.1.73.dylib" )

# Import target "mcl::mclbn384" for configuration "Release"
set_property(TARGET mcl::mclbn384 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(mcl::mclbn384 PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmclbn384.1.73.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libmclbn384.1.dylib"
  )

list(APPEND _cmake_import_check_targets mcl::mclbn384 )
list(APPEND _cmake_import_check_files_for_mcl::mclbn384 "${_IMPORT_PREFIX}/lib/libmclbn384.1.73.dylib" )

# Import target "mcl::mclbn384_256" for configuration "Release"
set_property(TARGET mcl::mclbn384_256 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(mcl::mclbn384_256 PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmclbn384_256.1.73.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libmclbn384_256.1.dylib"
  )

list(APPEND _cmake_import_check_targets mcl::mclbn384_256 )
list(APPEND _cmake_import_check_files_for_mcl::mclbn384_256 "${_IMPORT_PREFIX}/lib/libmclbn384_256.1.73.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
