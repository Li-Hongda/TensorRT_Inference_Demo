#include "common.h"
#include<dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

std::vector<std::string> ReadFolder(const std::string &image_path)
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

std::map<int, std::string> ReadImageNetLabel(const std::string &fileName)
{
    std::map<int, std::string> imagenet_label;
    std::ifstream file(fileName);
    if (!file.is_open())
    {
        std::cout << "read file error: " << fileName << std::endl;
    }
    std::string strLine;
    while (getline(file, strLine))
    {
        int pos1 = strLine.find(":");
        std::string first = strLine.substr(0, pos1);
        int pos2 = strLine.find_last_of("'");
        std::string second = strLine.substr(pos1 + 3, pos2 - pos1 - 3);
        imagenet_label.insert({atoi(first.c_str()), second});
    }
    file.close();
    return imagenet_label;
}

int CheckDir(const std::string &path){
    int ret;
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        if (S_ISDIR(st.st_mode)) {
            return 0;
        } else {
            return -1;
        }
    }
}