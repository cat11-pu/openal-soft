add_executable(OpenAL_Tests)

# Prefer a local googletest checkout (ALSOFT_GTEST_SOURCE_DIR); fall back to
# FetchContent only when none is provided.
if(DEFINED ALSOFT_GTEST_SOURCE_DIR AND EXISTS "${ALSOFT_GTEST_SOURCE_DIR}/CMakeLists.txt")
	add_subdirectory("${ALSOFT_GTEST_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/googletest"
		EXCLUDE_FROM_ALL)
else()
	include(FetchContent)
	FetchContent_Declare(
		googletest
		GIT_REPOSITORY https://github.com/google/googletest.git
		GIT_TAG        main
	)
	# For Windows: Prevent overriding the parent project's compiler/linker settings
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	FetchContent_MakeAvailable(googletest)
endif()

target_link_libraries(OpenAL_Tests PRIVATE
	OpenAL
	GTest::gtest_main
)

target_sources(OpenAL_Tests PRIVATE
example.t.cpp
source_group.t.cpp
)

target_include_directories(OpenAL_Tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(OpenAL_Tests PRIVATE AL_ALEXT_PROTOTYPES)

set_target_properties(OpenAL_Tests PROPERTIES
	CXX_STANDARD 20
	CXX_STANDARD_REQUIRED TRUE
)

add_executable(SourceGroupBench source_group_bench.cpp)
target_link_libraries(SourceGroupBench PRIVATE OpenAL)
target_include_directories(SourceGroupBench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(SourceGroupBench PRIVATE AL_ALEXT_PROTOTYPES)
set_target_properties(SourceGroupBench PROPERTIES
	CXX_STANDARD 20
	CXX_STANDARD_REQUIRED TRUE
)

# This needs to come last
include(GoogleTest)
gtest_discover_tests(OpenAL_Tests)
