# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/user/sof/software3/sof_cv

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/user/sof/software3/sof_cv/build

# Include any dependencies generated for this target.
include CMakeFiles/sof.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sof.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sof.dir/flags.make

CMakeFiles/sof.dir/src/sof.cpp.o: CMakeFiles/sof.dir/flags.make
CMakeFiles/sof.dir/src/sof.cpp.o: ../src/sof.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/user/sof/software3/sof_cv/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sof.dir/src/sof.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sof.dir/src/sof.cpp.o -c /home/user/sof/software3/sof_cv/src/sof.cpp

CMakeFiles/sof.dir/src/sof.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sof.dir/src/sof.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/user/sof/software3/sof_cv/src/sof.cpp > CMakeFiles/sof.dir/src/sof.cpp.i

CMakeFiles/sof.dir/src/sof.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sof.dir/src/sof.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/user/sof/software3/sof_cv/src/sof.cpp -o CMakeFiles/sof.dir/src/sof.cpp.s

# Object files for target sof
sof_OBJECTS = \
"CMakeFiles/sof.dir/src/sof.cpp.o"

# External object files for target sof
sof_EXTERNAL_OBJECTS =

sof: CMakeFiles/sof.dir/src/sof.cpp.o
sof: CMakeFiles/sof.dir/build.make
sof: CMakeFiles/sof.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/user/sof/software3/sof_cv/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable sof"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sof.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sof.dir/build: sof

.PHONY : CMakeFiles/sof.dir/build

CMakeFiles/sof.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sof.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sof.dir/clean

CMakeFiles/sof.dir/depend:
	cd /home/user/sof/software3/sof_cv/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/user/sof/software3/sof_cv /home/user/sof/software3/sof_cv /home/user/sof/software3/sof_cv/build /home/user/sof/software3/sof_cv/build /home/user/sof/software3/sof_cv/build/CMakeFiles/sof.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sof.dir/depend
