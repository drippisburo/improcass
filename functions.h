#pragma once
#include <vector>
#include <Eigen/Dense> 
#include <string> 


struct Point {
    int x;
    int y;
};

struct Blob {
    int area;
    int xmin, ymin, xmax, ymax;
    float mx, my;
    std::vector<Point> points; //to be removed but for visualization
};


struct Eigenstuff {
    Eigen::VectorXf eigvals;
    Eigen::MatrixXf eigvects;  
    Eigen::VectorXf mean;
    Eigen::MatrixXf Y;         // either after pca or after lda so it changes
};

struct Patch {
    std::vector<unsigned char> pixels;   // the rotated patch
    int label = -1;                      // 0 = clutter, 1 = person, 2 = cluster
    std::string image;                   // like "1_60_80_0_00640.jpg", for checking results
    std::string group;                   // like "1_60_80_0", for the folds
};

struct Box { int xmin, ymin, xmax, ymax; };

bool hotpixmask(float pix, float mean, float bright_threshold);
void makeimage(unsigned char *data, int w, int h, const char *filename);
std::vector<float> uchartofloat(std::vector<unsigned char> &data, int w, int h, double &sum_pix_frame);
std::vector<bool> floattobool(std::vector<float> &fdata, int w, int h, const double &mean_pix, const float bright_threshold);
std::vector<unsigned char> booltochar(std::vector<bool> &booldata, int w, int h);
std::vector<Blob> findblob(std::vector<bool> booldata, int w, int h, int r, int minarea, int maxarea);
void nicepatches(int size, std::vector<unsigned char> &imtofix, int half, int cy, int cx, int w, int h);
std::vector<std::vector<unsigned char>> rectpatchfromframe(std::vector<Blob> &blobs, std::vector<unsigned char> &data, int h, int w, int rad, int margin);
std::vector<unsigned char> visualizeblobs(std::vector<Blob> &blobs, int h, int w);
bool incircle(int x, int y, int size, int margin);
float patchangle(const std::vector<unsigned char> &patch, int size, int margin);
std::vector<unsigned char> rotatepatch(const std::vector<unsigned char> &patch, int size, int margin, float theta);
void patchrotator(std::vector<std::vector<unsigned char>> &patches, int size, int margin);
void plotpatches(std::vector<std::vector<unsigned char>> &patches, int rad, int margin, int size);
void plotrotatedpatches(std::vector<std::vector<unsigned char>> &patches, int rad, int margin, int size);
Eigenstuff eigenfinder(const std::vector<Patch> &patches, int size, int kept, int margin);
std::vector<std::vector<Patch>> imcategorizer(std::vector<int> camangle, std::vector<int> height, std::string folderim,  std::string folderxml, int &w, int &h, float bright_threshold, int minarea, int maxarea, int r, int rad, int margin, int size);
std::vector<Box> readboxes(const std::string &path, const std::string &want);   // in the header
void traintestbuckets(const std::vector<std::vector<int>> &classpercat, const std::vector<std::vector<Patch>> &allcat, std::vector<std::vector<Patch>> &buckets);
std::vector<Patch> combinebuckets(const std::vector<std::vector<Patch>> &buckets, const std::vector<int> &trainortest);
Eigenstuff LDA(Eigenstuff eigkept, std::vector<Patch> patches);
std::vector<unsigned char> mirrorpatch(const std::vector<unsigned char> &patch, int size, int margin, bool leftright);
std::vector<Eigenstuff> mdistclass(Eigenstuff eigklda, std::vector<Patch> patches);
double mahanobianscore(const Eigenstuff &eigclass, const Eigen::VectorXf &x, float bias);
std::vector<Eigenstuff> adjeigvals(std::vector<Eigenstuff> &eigstuff, int choice, int small);
Eigen::VectorXf zscore(Eigen::VectorXf x, int size, int margin);
