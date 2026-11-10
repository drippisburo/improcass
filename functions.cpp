#include <stdio.h>
#include <iostream>
#include <math.h>
#include "stb_image_write.h"
#include "stb_image.h"
#include "functions.h"
#include <vector>
#include <Eigen/Dense>
#include <filesystem> 
#include <format>
#include <algorithm>
#include <fstream>

using namespace std;
using namespace Eigen;
using namespace std::filesystem;


bool hotpixmask(float pix, float mean, float bright_threshold) {
    return (pix >= bright_threshold * mean) ? true : false;
}

void makeimage(unsigned char *data, int w, int h, const char *filename) {
    int ok = stbi_write_png(filename, w, h, 1, data, w);
    if (!ok) cout << "failed to write the imaggg" << filename << endl;
}

//from unchar to float vector
vector<float> uchartofloat( vector<unsigned char> &data, int w, int h, double &sum_pix_frame){ //need to give this as 0
    vector<float> fdata(w * h); 
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                fdata[y * w + x] = (float)data[y * w + x];
                sum_pix_frame += fdata[y * w + x];
            }
        }
        return fdata;
    }
        
//from fdata to masks for blobs
vector<bool> floattobool(vector<float> &fdata, int w, int h, const double &mean_pix, const float bright_threshold) {
    vector<bool> booldata(w * h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            booldata[y * w + x] = hotpixmask(fdata[y * w + x], mean_pix, bright_threshold);
        }
    }
    return booldata;
}

//from booldata to unchar
vector<unsigned char> booltochar( vector<bool> &booldata, int w, int h){
    vector<unsigned char> maskdata( w*h );
    for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                maskdata[y * w + x] = booldata[y * w + x] ? (unsigned char)255 : (unsigned char)0;
            }
        }
    return maskdata;
    }

vector<Blob> findblob(vector<bool> booldata, int w, int h, int r, int minarea, int maxarea) {
    vector<Blob> blobs;      // every blob found in the frame in here
    vector<int> todo;        // pixels of the current blob that still need a look around

    for (int y = 0; y < h; y++){
        for (int x = 0; x < w; x++){
            if (booldata[y*w + x]) {
                // a new blob starts here
                Blob hi;
                hi.area = 0;
                hi.xmin = w;  hi.ymin = h;
                hi.xmax = -1; hi.ymax = -1;
                double sumx = 0, sumy = 0;

                booldata[y*w + x] = false;    // off so not counted no more
                todo.push_back(y*w + x);
                //hi.points.push_back({ x , y }); //activate for blobs plotting

                while (!todo.empty()) {
                    int now = todo.back();
                    todo.pop_back();
                    int nowx = now % w;
                    int nowy = now / w;

                    // save the info of this pixel
                    hi.area += 1;
                    sumx += nowx;
                    sumy += nowy;
                    if (nowx < hi.xmin) hi.xmin = nowx;
                    if (nowx > hi.xmax) hi.xmax = nowx;
                    if (nowy < hi.ymin) hi.ymin = nowy;
                    if (nowy > hi.ymax) hi.ymax = nowy;

                    // look around: the x neighbours depending on r
                    for (int dy = -r; dy <= r; dy++){
                        for (int dx = -r; dx <= r; dx++){
                            int newx = nowx + dx;
                            int newy = nowy + dy;
                            if (newx >= 0 && newx < w && newy >= 0 && newy < h){
                                if (booldata[newy*w + newx]){
                                    booldata[newy*w + newx] = false;
                                    todo.push_back(newy*w + newx);
                                    //hi.points.push_back({ newx , newy }); //to activate to see blobs
                                }
                            }
                        }
                    }
                }
                if (hi.area < minarea || hi.area > maxarea) {continue;}  // ignore small blobs
                hi.mx = sumx / hi.area;
                hi.my = sumy / hi.area;
                blobs.push_back(hi);
            }
        }
    }
    return blobs;
}

// make patches nice
void nicepatches( int size, vector<unsigned char>& imtofix, int half, int cy, int cx, int w, int h){
    int ystart = max(half - cy, 0);
    int yend   = size - max(cy + half - h, 0);
    int xstart = max(half - cx, 0);
    int xend   = size - max(cx + half - w, 0);

    // mean of the real pixels only
    int sum = 0;
    for (int y = ystart; y < yend; y++) {
        for (int x = xstart; x < xend; x++) {
            sum += imtofix[y*size + x];
        }
    }
    unsigned char mean = sum / ((yend - ystart) * (xend - xstart));
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            if (y < ystart || y >= yend || x < xstart || x >= xend) {
                imtofix[y*size + x] = mean;
            }
        }
    }
}

// patches centered on the blobs, at the edge the edge value fets extended
// patches cut as if the drone flew at the reference altitude: k = altref / alt image pixels per patch pixel
vector<vector<unsigned char>> scaledpatchfromframe(vector<Blob> &blobs, vector<unsigned char> &data, int h, int w, int rad, int margin, float k) {
    vector<vector<unsigned char>> patches;
    int half = rad + margin;
    int size = 2 * half;
    for (const Blob &blo : blobs) {
        vector<unsigned char> patch(size * size);
        for (int v = 0; v < size; v++) {
            for (int u = 0; u < size; u++) {
                float x = blo.mx + (u - half) * k;      // float centroid,unrounded
                float y = blo.my + (v - half) * k;
                x = min(max(x, 0.f), w - 1.001f);       // if outside the frame border pixel is repeted
                y = min(max(y, 0.f), h - 1.001f);
                int x0 = (int)x, y0 = (int)y;
                float fx = x - x0, fy = y - y0;
                float val = (1-fx) * (1-fy) * data[y0 * w + x0]
                          + fx     * (1-fy) * data[y0 * w + x0 + 1]
                          + (1-fx) * fy     * data[(y0 + 1) * w + x0]
                          + fx     * fy     * data[(y0 + 1) * w + x0 + 1];
                patch[v * size + u] = (unsigned char)(val + 0.5f);
            }
        }
        patches.push_back(patch);
    }
    return patches;
}


//this is for visualizing the blobs only
vector<unsigned char> visualizeblobs(vector<Blob> &blobs, int h, int w) {
    vector<unsigned char> newim( h*w, 0);
    for (const Blob &blo : blobs){
        for (const Point &p : blo.points) {
            newim[p.y*w + p.x] = 255;
        }
    }
    return newim;
}
// check: is it inside the circle
bool incircle(int x, int y, int size, int margin) {
    int half = size / 2;                // the blob centre is at (half, half)
    int R = half - margin;
    float dx = x - half;
    float dy = y - half;
    return dx*dx + dy*dy <= R*R;
}

// finding the PCA to rotate at the angle decided in radians
float patchangle(const vector<unsigned char> &patch, int size, int margin) { 
    float mx = size / 2;
    float my = mx;
    double mu20 = 0, mu02 = 0, mu11 = 0;
    float sum = 0; //for first finding the mean
    int count = 0;
    for (int i = 0; i < size*size; i++){
        if (incircle(i%size, i/size, size, margin)) {
             sum += (int)patch[i];
             count++;
        }
    }
    float mean = sum/(count > 0 ? count : 1);

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            if (incircle(x, y, size, margin)) {
                float bright_adj = (int)patch[y*size +x] - mean;
                float dx = (x- mx);
                float dy = (y- my);
                mu20 += dx * dx*bright_adj;
                mu02 += dy * dy*bright_adj;
                mu11 += dx * dy*bright_adj;
            }
        }
    }
    return 0.5 * atan2(2 * mu11, mu20 - mu02);   // radians
}

// rotate the patch so the long axis ends up vertical, theta from patchangle
vector<unsigned char> rotatepatch(const vector<unsigned char> &patch, int size, int margin, float theta) { // margin must be >= 2
    vector<unsigned char> rotated(size * size, 0);
    int half = size / 2;
    float t = theta - 1.5707963f;    // minus 90 degrees, in radians
    float c = cos(t), s = sin(t);
    for (int v = 0; v < size; v++) {
        for (int u = 0; u < size; u++) {
            if (!incircle(u, v, size, margin)) continue;   // corners stay black

            // where exactly in the original patch it come from
            float du = u - half;
            float dv = v - half;
            float x = half + c * du - s * dv;
            float y = half + s * du + c * dv;

            // blend the 4 nearest pixels from the original weighted on distance
            int x0 = (int)floor(x);
            int y0 = (int)floor(y);
            float fx = x - x0;
            float fy = y - y0;
            float val = (1-fx) * (1-fy) * patch[y0 * size + x0]
                      + fx     * (1-fy) * patch[y0 * size + x0 + 1]
                      + (1-fx) * fy     * patch[(y0 + 1) * size + x0]
                      + fx     * fy     * patch[(y0 + 1) * size + x0 + 1];

            rotated[v * size + u] = (unsigned char)(val + 0.5f);   // rounded 
        }
    }
    return rotated;
}

//rotating the patches based on their highest variance axis
void patchrotator(vector<vector<unsigned char>> &patches, int size, int margin){
        int counter = 0;
            for (vector<unsigned char>& patch : patches) {
                float angle = patchangle( patch, size, margin);
                vector<unsigned char> rotated_patch = rotatepatch( patch, size, margin, angle);                patches[counter] = rotated_patch;
                counter++;
            }
        }



//plot patches, ofc optional
void plotpatches(vector<vector<unsigned char>>& patches, int rad, int margin, int size){
    int count = 0;
    for (vector<unsigned char> patch : patches) {
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                if (!incircle(x, y, size , margin)) {
                    patch[ y*size + x] = (unsigned char)0;
                }
            }
        }
        makeimage(patch.data(),size , size, ("outputpatchbefore_" + to_string(count) + ".png").c_str());
        count++;
    }
}

//plot ROTATED patches, ofc optional
void plotrotatedpatches(vector<vector<unsigned char>>& patches, int rad, int margin, int size){
    int count = 0;
    for (vector<unsigned char> patch : patches) {
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                if (!incircle(x, y, size , margin)) {
                    patch[ y*size + x] = (unsigned char)0;
                }
            }
        }
        makeimage(patch.data(),size , size, ("outputpatchROTATED_" + to_string(count) + ".png").c_str());
        count++;
    }
}


//finds and gives eigenvalues and vec of all images given
Eigenstuff eigenfinder(const vector<Patch> &patches, int size, int kept, int margin) {
    //first normalizing by finding the mean across all patches
    int nim = patches.size();
    int pixels = size * size;
    MatrixXf X(pixels, nim);
    Eigenstuff eigkept;
    eigkept.mean = VectorXf::Zero(pixels);

    for ( int nofpatch = 0; nofpatch < patches.size(); nofpatch++ ) {
        X.col(nofpatch) = Map<const Matrix<unsigned char, Dynamic, 1>>(patches[nofpatch].pixels.data(), pixels).cast<float>();
        X.col(nofpatch) = zscore(X.col(nofpatch), size, margin);
        eigkept.mean += X.col(nofpatch);
    }

    eigkept.mean /= (float)nim;
    X.colwise() -= eigkept.mean;
    
    //finally all the images have been put into a matrix and they have been averaged out
    MatrixXf S;
    if (nim > pixels) S = X * X.transpose() / (nim - 1);   // pixels x pixels
    else              S = X.transpose() * X / (nim - 1);   // nim x nim
    SelfAdjointEigenSolver<MatrixXf> eigenstuff(S);

    //keeping the right eigenvalues and vectors
    kept = min(kept, min(pixels, nim - 1));
    eigkept.eigvals = eigenstuff.eigenvalues().tail(kept);
    if (nim > pixels) {
        eigkept.eigvects = eigenstuff.eigenvectors().rightCols(kept);
    } else {
        eigkept.eigvects = X * eigenstuff.eigenvectors().rightCols(kept);
        eigkept.eigvects.colwise().normalize();
    }
    eigkept.Y = eigkept.eigvects.transpose() * X;

    return eigkept;

}

//reads the xml file and prints out the location of the category box
vector<Box> readboxes(const string &path, const string &categor) {
    vector<Box> boxes;
    ifstream file(path);
    if (!file) { cout << "could not open xml :(" << path << endl; return boxes; }

    string line, name;
    Box b;
    while (getline(file, line)) {
        char text[31];
        if (sscanf(line.c_str(), " <name>%30[^<]", text) == 1) name = text;
        sscanf(line.c_str(), " <xmin>%d", &b.xmin);
        sscanf(line.c_str(), " <ymin>%d", &b.ymin);
        sscanf(line.c_str(), " <xmax>%d", &b.xmax);
        if (sscanf(line.c_str(), " <ymax>%d", &b.ymax) == 1 && name == categor) boxes.push_back(b);
    }
    return boxes;
}

int assignclass( Blob &blob, string folderxml, string image) {
    int dayornight = 1; // only working in the night
    int alti, anglei, numi;
    if (sscanf(image.c_str(), "%d_%d_%d_0_%d.jpg", &dayornight, &alti, &anglei, &numi) != 4){cout<<"classification had problems"<<endl;}
    string xml = format("{}_{}_{}_0_{:05}.xml", dayornight, alti, anglei, numi);
    string path = folderxml + xml;
    string person = "Person";
    int pcount = 0;
    vector<Box> boxes = readboxes( path, person);
    for ( const Box &box : boxes) {
        int cx = (box.xmin + box.xmax) / 2;
        int cy = (box.ymin + box.ymax) / 2;
        if (cx >= blob.xmin && cx <= blob.xmax && cy >= blob.ymin && cy <= blob.ymax) pcount++;
    }
    if (pcount == 1) {return 1;}     //class person
    else if (pcount == 0) {return 0;}//class clutter
    else {return 2;}                 //class cluster
}


//divide into all possible (night) categories
vector<vector<Patch>> imcategorizer(vector<int> camangle, vector<int> height, string folderim, string folderxml, int &w, int &h, float bright_threshold, int minarea, int maxarea, int r, int rad, int margin, int size) {

    // one empty category for every angle + height combination
    vector<vector<Patch>> allcategpatches(camangle.size() * height.size());
    int alti, anglei, numi;

        //getting images from files
    vector<string> names;
    for (const auto &entry : directory_iterator(folderim)) {
        names.push_back(entry.path().filename().string());
    }
    sort(names.begin(), names.end());

    //groupin images from height and angles
    for (const string &image : names) {
        int night = -1;
        if (sscanf(image.c_str(), "%d_%d_%d_0_%d.jpg", &night, &alti, &anglei, &numi) != 4) continue; //onlyn night
        if (night != 1) continue;
        int c_ang = 0;
        for (int angle : camangle) {
            int c_height = 0;
            for (int alt : height) {
                if (angle == anglei && alt == alti) {
                    string imname = folderim + image;
                    unsigned char *im = stbi_load(imname.c_str(), &w, &h, NULL, 1);
                    if (im) {

                        double sum_pix_frame = 0;
                        vector<unsigned char> img(im, im + w * h);
                        vector<float> fdata = uchartofloat( img, w, h, sum_pix_frame);
                        double mean_pix = sum_pix_frame / (w * h);
                        vector<bool> booldata = floattobool(fdata, w, h, mean_pix, bright_threshold);
                        //makeimage( booltochar( booldata, w, h).data(), w, h, "outputmask.png");

                        //now onto the blobs
                        vector<Blob> blobs = findblob( booldata, w, h, r, minarea, maxarea);
                        //makeimage(visualizeblobs(blobs, h, w).data(), w, h, "outputblobs.png"); //blobs better

                        //getting patches from each frame
                        float altscale = 100.0 / alt;
                        vector<vector<unsigned char>> patches = scaledpatchfromframe(blobs, img, h, w, rad, margin, altscale);
                        //plotting patches before they are rotated
                        //plotpatches( patches, rad,  margin, size);

                        patchrotator(patches, size, margin);
                        //plotting patches after they are rotated
                        //plotrotatedpatches( patches, rad,  margin, size);

                        //put the nice and tidy information about name, patch and group that will be outputted
                        for (int pnum = 0; pnum < patches.size(); pnum++) {
                            Patch p;
                            p.pixels = patches[pnum];
                            p.label = assignclass(blobs[pnum], folderxml, image);
                            p.image = image;
                            p.group = image.substr(0, image.rfind('_'));    //groupname from alt and angle
                            allcategpatches[c_ang * height.size() + c_height].push_back(p);
                        }


                        stbi_image_free(im);
                    } else {
                        cout << "could not load " << image << endl;
                    }
                }
                c_height++;
            }
            c_ang++;
        }
    }
    return allcategpatches;
}

//returns which category has the most clusters
int catmaxclust(vector<vector<int>> &classpercat){
    int count = 0;
    int pos = 0;
    int max = -1;
    for ( vector<int> category : classpercat){
        if (category[2] > max){max = category[2];pos = count;}
        count++;
    }
    return pos; //this is the position of the category with the max clusters
}

//returns which bucket has the less clusters
int buckleastclust(vector<vector<vector<int>>> bucketscat){
    int least = 0;
    int n_bucket = 0;
    int pos = 0;
    for ( vector<vector<int>> bucket : bucketscat ){
        int n_clust_buck = 0;
        for ( vector<int> patch : bucket){
            n_clust_buck += patch[2];
        }
        if (n_bucket == 0 || n_clust_buck < least) {least = n_clust_buck; pos = n_bucket;}
        n_bucket++;
    }
    return pos;
}

// this allocates them categories in the rigth "bucket"
// it is a train test split that allocates a group in a bucket
// based on what is the bucket with less clusters, since 
// there is scarcity of clusters
void traintestbuckets(const vector<vector<int>> &classpercat,const vector<vector<Patch>> &allcat, vector<vector<Patch>>& buckets){
    int counter = 0;
    
    vector<vector<int>> copyclasspercat = classpercat;
    vector<vector<vector<int>>> bucketscat(buckets.size()); // has number of categories per each frame in their bucket
                                                       // it is a vector of n buckets holding in each a frame and its categories number
    while (counter < classpercat.size()){
        int maxclustleft = catmaxclust(copyclasspercat);
        copyclasspercat[maxclustleft][2] = -1;       // removed from the search

        int leastclustbucket = buckleastclust(bucketscat); //which bucket had the least clusters
        bucketscat[leastclustbucket].push_back(classpercat[maxclustleft]); //putting the category in the bucket

        for (const Patch &patch : allcat[maxclustleft]){  //going through all patchesfrom one category
            buckets[leastclustbucket].push_back(patch);
        }

        counter++;

    }
}

//combine the buckets into train or test
vector<Patch> combinebuckets(const vector<vector<Patch>> &buckets, const vector<int> &trainortest){
    vector<Patch> combined;
    for ( int bucket : trainortest ) {
        for ( const Patch &p : buckets[bucket]){
            combined.push_back(p);
        }
    }
    return combined;
}

//LDA

Eigenstuff LDA(Eigenstuff eigkept, vector<Patch> patches){
    VectorXf totmean = VectorXf::Zero(eigkept.Y.rows());
    MatrixXf meanclass = MatrixXf::Zero(eigkept.Y.rows(), 3);
    int people = 0;
    int cluster = 0;
    int clutter = 0;
    int tot = patches.size();
    for ( int nofpatch = 0; nofpatch < patches.size(); nofpatch++ ) {
        totmean += eigkept.Y.col(nofpatch);
        if (patches[nofpatch].label == 1){
            meanclass.col(1) += eigkept.Y.col(nofpatch);
            people++;
        }
        else if (patches[nofpatch].label == 2){
            meanclass.col(2) += eigkept.Y.col(nofpatch);
            cluster++;
        }
        else {
            meanclass.col(0) += eigkept.Y.col(nofpatch);
            clutter++;
        }
    }
    totmean /= (float)patches.size();
    meanclass.col(0) /= (float)clutter;
    meanclass.col(1) /= (float)people;
    meanclass.col(2) /= (float)cluster;
    //all means found
    MatrixXf classone(eigkept.Y.rows(), people);
    MatrixXf classtwo(eigkept.Y.rows(), cluster);
    MatrixXf classzero(eigkept.Y.rows(), clutter);
    int pcount = 0;
    int ccount = 0; //this is for clusters
    int zcount = 0; //clutter
    for ( int nofpatch = 0; nofpatch < patches.size(); nofpatch++ ) {
        if (patches[nofpatch].label == 1){
            classone.col(pcount) = eigkept.Y.col(nofpatch) - meanclass.col(1);
            pcount++;
        }
        else if (patches[nofpatch].label == 2){
            classtwo.col(ccount) = eigkept.Y.col(nofpatch) - meanclass.col(2);
            ccount++;
        }
        else {
            classzero.col(zcount) = eigkept.Y.col(nofpatch) - meanclass.col(0);
            zcount++;
        }
    }

    //now onto the within class covariance
    MatrixXf Swone = (classone * classone.transpose()) / eigkept.Y.cols();
    MatrixXf Swtwo = (classtwo * classtwo.transpose()) / eigkept.Y.cols();
    MatrixXf Swzero = (classzero * classzero.transpose()) / eigkept.Y.cols();
    
    //the final within class cov matrix
    MatrixXf Sw = Swone + Swtwo + Swzero;

    //now the between class cov matrix
    MatrixXf Sbone = people/(float)tot * (( meanclass.col(1) - totmean ) * ( meanclass.col(1) - totmean ).transpose());
    MatrixXf Sbtwo = cluster/(float)tot * (( meanclass.col(2) - totmean ) * ( meanclass.col(2) - totmean ).transpose());
    MatrixXf Sbzero = clutter/(float)tot * (( meanclass.col(0) - totmean ) * ( meanclass.col(0) - totmean ).transpose());

    MatrixXf Sb = Sbone + Sbtwo + Sbzero; //final final sb

    Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXf> ges(Sb, Sw);

    Eigenstuff eigkeptlda;
    eigkeptlda.eigvals = ges.eigenvalues().tail(2);          // 2 largest, smallest first
    eigkeptlda.eigvects = ges.eigenvectors().rightCols(2);   // eigenvector i is column i
    eigkeptlda.mean = totmean;
    eigkeptlda.Y = eigkeptlda.eigvects.transpose() * eigkept.Y;
    // The largest eigenvalues are at the end, so take the last C-1 columns:
    MatrixXf W = eigkeptlda.eigvects.rightCols(3 - 1);   // 3 = number of classes

    return eigkeptlda;
    
}

//flips each patch for better training
vector<unsigned char> mirrorpatch(const vector<unsigned char> &patch, int size, int margin, bool leftright) {
    vector<unsigned char> mirrored(size * size, 0);
    int half = size / 2;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            if (!incircle(x, y, size, margin)) continue;   // corners stay black
            int sx = leftright ? 2 * half - x : x;
            int sy = leftright ? y : 2 * half - y;
            mirrored[y * size + x] = patch[sy * size + sx];
        }
    }
    return mirrored;
}

//mahanobian distance
vector<Eigenstuff> mdistclass(Eigenstuff eigklda, vector<Patch> patches){
    MatrixXf meanclass = MatrixXf::Zero(eigklda.Y.rows(), 3);
    int people = 0;
    int cluster = 0;
    int clutter = 0;
    int tot = patches.size();
    for ( int nofpatch = 0; nofpatch < patches.size(); nofpatch++ ) {
        
        if (patches[nofpatch].label == 1){
            meanclass.col(1) += eigklda.Y.col(nofpatch);
            people++;
        }
        else if (patches[nofpatch].label == 2){
            meanclass.col(2) += eigklda.Y.col(nofpatch);
            cluster++;
        }
        else {
            meanclass.col(0) += eigklda.Y.col(nofpatch);
            clutter++;
        }
    }
    meanclass.col(0) /= (float)clutter;
    meanclass.col(1) /= (float)people;
    meanclass.col(2) /= (float)cluster;
    //all means found
    MatrixXf classone(eigklda.Y.rows(), people);
    MatrixXf classtwo(eigklda.Y.rows(), cluster);
    MatrixXf classzero(eigklda.Y.rows(), clutter);
    int pcount = 0;
    int ccount = 0; //this is for clusters
    int zcount = 0; //clutter
    for ( int nofpatch = 0; nofpatch < patches.size(); nofpatch++ ) {
        if (patches[nofpatch].label == 1){
            eigklda.Y.col(nofpatch) -= meanclass.col(1);
            classone.col(pcount) = eigklda.Y.col(nofpatch);
            pcount++;
        }
        else if (patches[nofpatch].label == 2){
            eigklda.Y.col(nofpatch) -= meanclass.col(2);
            classtwo.col(ccount) = eigklda.Y.col(nofpatch);
            ccount++;
        }
        else {
            eigklda.Y.col(nofpatch) -= meanclass.col(0);
            classzero.col(zcount) = eigklda.Y.col(nofpatch);
            zcount++;
        }
    }
    //nex step is to find the covariance matrix of each class and then 
        //nex step is to find the covariance matrix of each class and then its eigenvalues and eigenvectors
    MatrixXf Cone = (classone * classone.transpose()) / ((float)people - 1);
    MatrixXf Ctwo = (classtwo * classtwo.transpose()) / ((float)cluster - 1);
    MatrixXf Czero = (classzero * classzero.transpose()) / ((float)clutter - 1);

    SelfAdjointEigenSolver<MatrixXf> esone(Cone);
    SelfAdjointEigenSolver<MatrixXf> estwo(Ctwo);
    SelfAdjointEigenSolver<MatrixXf> eszero(Czero);

    Eigenstuff eigone;
    Eigenstuff eigtwo;
    Eigenstuff eigzero;

    eigone.mean = meanclass.col(1);
    eigtwo.mean = meanclass.col(2);
    eigzero.mean = meanclass.col(0);

    eigone.eigvals = esone.eigenvalues();      // smallest first
    eigtwo.eigvals = estwo.eigenvalues();
    eigzero.eigvals = eszero.eigenvalues();

    eigone.eigvects = esone.eigenvectors();    // column k goes with eigvals(k)
    eigtwo.eigvects = estwo.eigenvectors();
    eigzero.eigvects = eszero.eigenvectors();

    return {eigzero, eigone, eigtwo};  
}

//optional to keep either the average of the smaller eigenvalues or
//to assign the smaller eigenvalues the biggest between them
//choice 0 is using the bigger eigval, 1 is their average
vector<Eigenstuff> adjeigvals(vector<Eigenstuff> &eigstuff, int choice, int notsmall){
    if (choice != 0 && choice != 1) return eigstuff;
    for (int i = 0; i < 3; i++){
        int m = eigstuff[i].eigvals.size();
        if (choice == 0){
            for (int k = 0; k < m - notsmall - 1; k++) eigstuff[i].eigvals(k) = eigstuff[i].eigvals(m - notsmall - 1);
        }
        else if (choice == 1){
            double ave = 0.0;
            for (int k = 0; k < m - notsmall; k++) ave += eigstuff[i].eigvals(k);
            ave /= (m - notsmall);
            for (int k = 0; k < m - notsmall; k++) eigstuff[i].eigvals(k) = ave;
        }
    }
    return eigstuff;
}

double mahanobianscore(const Eigenstuff &eigclass, const VectorXf &x, float bias){
    VectorXf xave = x - eigclass.mean;
    VectorXf z = eigclass.eigvects.transpose() * xave;
    double sum = 0;
    for (int i = 0; i < eigclass.eigvals.size(); i++){
        sum += z(i)*z(i)/eigclass.eigvals(i);
    }
    return -1/2.0 * (sum) + bias;
}

// z-score one patch inside the circle, outside stays 0
VectorXf zscore(VectorXf x, int size, int margin) {
    double s = 0, s2 = 0;
    int n = 0;
    for (int i = 0; i < size * size; i++)
        if (incircle(i % size, i / size, size, margin)) { s += x(i); s2 += (double)x(i) * x(i); n++; }
    double mu = s / n;
    double sd = sqrt(max(s2 / n - mu * mu, 0.0));
    if (sd < 1) sd = 1;      // nearly flat patch: don't divide by ~0
    for (int i = 0; i < size * size; i++)
        x(i) = incircle(i % size, i / size, size, margin) ? (x(i) - mu) / sd : 0;
    return x;
}

// nnclosest neighbour, output is an int = 0 clutter, 1 person, 2 clutter
int nnclass( const VectorXf &Ztest, const MatrixXf &Xtest,const vector<Patch> &train_data, int n){
    vector<int> zclasses;
    
    for (int colx = 0; colx <Xtest.cols(); colx++){
        int distsq = (Ztest - Xtest.col(colx)).squaredNorm();
        zclasses.push_back(distsq);
    }
    vector<int> zcl_copy = zclasses;
    vector<int> scores = {0 , 0 , 0};
    vector<float> closest = {INFINITY, INFINITY, INFINITY};
    for ( int i = 0; i<n ; i++){
        auto it = min_element(zcl_copy.begin(), zcl_copy.end());
        int idx = (int)(it - zcl_copy.begin());
        zcl_copy[idx] = INFINITY;
        if (train_data[idx].label == 1) {
            scores[1]++;
            if ( zclasses[idx] < closest[1]) closest[1] =zclasses[idx];
        }
        else if(train_data[idx].label == 2) {
            scores[2]++;
            if ( zclasses[idx] < closest[2]) closest[2] =zclasses[idx];
        }
        else {
            scores[0]++;
            if ( zclasses[idx] < closest[0]) closest[0] =zclasses[idx];
        }
    }
    int max = -1;
    int foundcl = -1;
    for ( int i = 0; i < 3; i++){
        if (scores[i] > max) {max = scores[i]; foundcl = i;}
        if ( scores[i] == max) {
            int min = INFINITY;
            int where = -1;
            for ( int j = 0; j < i; j++){
                if ( scores[j] == max){
                    if (closest[j] < min){ min = closest[j]; where = j;}
                }
            }
            foundcl = where;
        }
    }
    return foundcl;
}
