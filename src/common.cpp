#include "common.h"
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>


bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
	if (code != cudaSuccess) {
		const char* err_name = cudaGetErrorName(code);
		const char* err_message = cudaGetErrorString(code);
		printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
		return false;
	}
	return true;
}

std::vector<std::string> get_names(const std::string &image_path)
{
    std::vector<std::string> image_names;
    auto dir = opendir(image_path.c_str());

    if ((dir) != nullptr)
    {
        struct dirent *entry;
        entry = readdir(dir);
        while (entry)
        {
            auto temp = image_path + "/" + entry->d_name;
            if (strcmp(entry->d_name, "") == 0 || strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            {
                entry = readdir(dir);
                continue;
            }
            image_names.emplace_back(temp);
            entry = readdir(dir);
        }
    }
    return image_names;
}


std::string replace(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

static bool isPathExists(const std::string& path){
    struct stat buffer;
    return (stat (path.c_str(), &buffer) == 0);
}

static bool isFile(const std::string& filename) {
    struct stat buffer;
    return S_ISREG(buffer.st_mode);
}
 
static bool isDirectory(const std::string& filefodler) {
    struct stat buffer;
    return S_ISDIR(buffer.st_mode);
}

static void createDirectory(const std::string &path) {
    uint32_t pathLen = path.length();

    char tmpDirPath[256] = { 0 };
    for (uint32_t i = 0; i < pathLen; ++i) {
        tmpDirPath[i] = path[i];
        if (tmpDirPath[i] == '\\' || tmpDirPath[i] == '/') {
            if (access(tmpDirPath, 0) != 0) {
                int32_t ret = mkdir(tmpDirPath, 00700);
            }
        }
    }
}

int check_dir(const std::string & path, const bool is_mkdir) noexcept {
    DIR *p_dir;
    struct dirent *entry;
    if(isPathExists(path)){
        if(isDirectory(path)){
            if((p_dir = opendir(path.c_str())) == NULL ) {  
                std::cout << "Opendir error: " << strerror(errno) << std::endl;
                return -1;  
            }  
    
            while((entry = readdir(p_dir)) != NULL){
                std::string file_name = path + "/" + entry->d_name;
                if((0 != strcmp(entry->d_name, ".")) && (0 != strcmp(entry->d_name, "..")))
                {
                    remove(file_name.c_str());
                }
            }
            closedir(p_dir);
            rmdir(path.c_str());
            mkdir(path.c_str(), 00700);
        }
    } else {
        // mkdir(path.c_str(), 00700);
        createDirectory(path);
    }
}