rmdir /s /q build
mkdir build
cd build
cmake .. -G "Visual Studio 15 2017 Win64" -DOpenCV_DIR="D:\opencv\opencv_450ht\install"
pause