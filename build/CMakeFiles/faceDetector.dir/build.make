# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mano/MyStuff/udacity/CppND/CppND-CapstoneProject/FaceFilterApp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mano/MyStuff/udacity/CppND/CppND-CapstoneProject/FaceFilterApp/build

# Include any dependencies generated for this target.
include CMakeFiles/faceDetector.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/faceDetector.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/faceDetector.dir/flags.make

CMakeFiles/faceDetector.dir/src/faceDetector.cpp.o: CMakeFiles/faceDetector.dir/flags.make
CMakeFiles/faceDetector.dir/src/faceDetector.cpp.o: ../src/faceDetector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mano/MyStuff/udacity/CppND/CppND-CapstoneProject/FaceFilterApp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/faceDetector.dir/src/faceDetector.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/faceDetector.dir/src/faceDetector.cpp.o -c /home/mano/MyStuff/udacity/CppND/CppND-CapstoneProject/FaceFilterApp/src/faceDetector.cpp

CMakeFiles/faceDetector.dir/src/faceDetector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/faceDetector.dir/src/faceDetector.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mano/MyStuff/udacity/CppND/CppND-CapstoneProject/FaceFilterApp/src/faceDetector.cpp > CMakeFiles/faceDetector.dir/src/faceDetector.cpp.i

CMakeFiles/faceDetector.dir/src/faceDetector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/faceDetector.dir/src/faceDetector.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mano/MyStuff/udacity/CppND/CppND-CapstoneProject/FaceFilterApp/src/faceDetector.cpp -o CMakeFiles/faceDetector.dir/src/faceDetector.cpp.s

CMakeFiles/faceDetector.dir/src/main.cpp.o: CMakeFiles/faceDetector.dir/flags.make
CMakeFiles/faceDetector.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mano/MyStuff/udacity/CppND/CppND-CapstoneProject/FaceFilterApp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/faceDetector.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/faceDetector.dir/src/main.cpp.o -c /home/mano/MyStuff/udacity/CppND/CppND-CapstoneProject/FaceFilterApp/src/main.cpp

CMakeFiles/faceDetector.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/faceDetector.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mano/MyStuff/udacity/CppND/CppND-CapstoneProject/FaceFilterApp/src/main.cpp > CMakeFiles/faceDetector.dir/src/main.cpp.i

CMakeFiles/faceDetector.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/faceDetector.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mano/MyStuff/udacity/CppND/CppND-CapstoneProject/FaceFilterApp/src/main.cpp -o CMakeFiles/faceDetector.dir/src/main.cpp.s

# Object files for target faceDetector
faceDetector_OBJECTS = \
"CMakeFiles/faceDetector.dir/src/faceDetector.cpp.o" \
"CMakeFiles/faceDetector.dir/src/main.cpp.o"

# External object files for target faceDetector
faceDetector_EXTERNAL_OBJECTS =

faceDetector: CMakeFiles/faceDetector.dir/src/faceDetector.cpp.o
faceDetector: CMakeFiles/faceDetector.dir/src/main.cpp.o
faceDetector: CMakeFiles/faceDetector.dir/build.make
faceDetector: /usr/local/lib/libopencv_dnn.so.4.1.0
faceDetector: /usr/local/lib/libopencv_gapi.so.4.1.0
faceDetector: /usr/local/lib/libopencv_ml.so.4.1.0
faceDetector: /usr/local/lib/libopencv_objdetect.so.4.1.0
faceDetector: /usr/local/lib/libopencv_photo.so.4.1.0
faceDetector: /usr/local/lib/libopencv_stitching.so.4.1.0
faceDetector: /usr/local/lib/libopencv_video.so.4.1.0
faceDetector: /usr/local/lib/libopencv_calib3d.so.4.1.0
faceDetector: /usr/local/lib/libopencv_features2d.so.4.1.0
faceDetector: /usr/local/lib/libopencv_flann.so.4.1.0
faceDetector: /usr/local/lib/libopencv_highgui.so.4.1.0
faceDetector: /usr/local/lib/libopencv_videoio.so.4.1.0
faceDetector: /usr/local/lib/libopencv_imgcodecs.so.4.1.0
faceDetector: /usr/local/lib/libopencv_imgproc.so.4.1.0
faceDetector: /usr/local/lib/libopencv_core.so.4.1.0
faceDetector: CMakeFiles/faceDetector.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mano/MyStuff/udacity/CppND/CppND-CapstoneProject/FaceFilterApp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable faceDetector"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/faceDetector.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/faceDetector.dir/build: faceDetector

.PHONY : CMakeFiles/faceDetector.dir/build

CMakeFiles/faceDetector.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/faceDetector.dir/cmake_clean.cmake
.PHONY : CMakeFiles/faceDetector.dir/clean

CMakeFiles/faceDetector.dir/depend:
	cd /home/mano/MyStuff/udacity/CppND/CppND-CapstoneProject/FaceFilterApp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mano/MyStuff/udacity/CppND/CppND-CapstoneProject/FaceFilterApp /home/mano/MyStuff/udacity/CppND/CppND-CapstoneProject/FaceFilterApp /home/mano/MyStuff/udacity/CppND/CppND-CapstoneProject/FaceFilterApp/build /home/mano/MyStuff/udacity/CppND/CppND-CapstoneProject/FaceFilterApp/build /home/mano/MyStuff/udacity/CppND/CppND-CapstoneProject/FaceFilterApp/build/CMakeFiles/faceDetector.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/faceDetector.dir/depend

