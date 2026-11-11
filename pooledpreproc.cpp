#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include "functions.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;
using namespace Eigen;   

int main() {
    int LDAPCA = 1; // 1 means lda is applied, 0 means only pca
    float bright_threshold = 1.52f; //this times mean frame is threshold for meaningful brightness
    int minarea = 12; // blobs smaller ignored 
    int maxarea = 500; // blobs larger ignored
    int r = 6; // radius for blob creation 
    int rad = 16; // radius for patch creation
    int margin = 3; // margin for patch creation
    int kept = 40; // how many eigenvectors (dimensions) are kept
    int w = 0, h = 0;
    double sum_pix_frame = 0;
    int size = 2 * (rad + margin);
    vector<int> height = {60, 70, 80, 90, 100, 110, 120, 130};
    vector<int> camangle = {70,80,90}; //choose any angle multiple of 10 between 30 and 90 
    int n_buckets = 5;
    int notsmall = 20; //how many eigvalues are not consider small
    int choice_eigvals = 0; // 0 means keep the biggest eigvals from the smaller ones
                            // 1 means keep the average
    int classifier = 1; // 0 = Mahalanobis, 1 = k-NN
    int k = 5;          // k for k nearest neighbour

    /*string image = "1_60_80_0_00715.jpg";*/ // for specific one image
    string folderim = "./normal_xml/JPEGImages/";
    string folderxml = "./normal_xml/Annotations/";
    
    //from images to ordered patches, the one below has
    //specific combinations of height and angles at the
    //position angle_p * height_size() + height_p
    vector<vector<Patch>> allcat = imcategorizer(camangle, height, folderim, folderxml, w, h, bright_threshold, minarea, maxarea, r, rad, margin, size);
    //now making a vector with the same mapping as the
    //one above to count each category's number of categ
    vector<vector<int>> classpercat(allcat.size(), vector<int>(3, 0));
    
    int num_cat = 0;
    for (const vector<Patch> &category : allcat) {
        for (const Patch &p : category) {
            classpercat[num_cat][p.label]++; //positions are 0 clutter,1 person and 2 cluster
        }
        
        cout << " height "<< height[num_cat%height.size()]
        << "meters and angle " << camangle[num_cat/height.size()];
        int i = 0;
        for (int count : classpercat[num_cat]) {
            cout << "category " << i << " = " << count << "  ";
            i++;
        }
        cout << endl;
        num_cat++;
    }

    //now making the (5) sorted "buckets", some to use on train some on test
    vector<vector<Patch>> buckets(n_buckets);
    traintestbuckets(classpercat, allcat, buckets);
    // ---- sweep settings ----
    vector<int> dlist = {1, 2, 3, 5, 10, 20, 40, 80, 150};   // PCA dimensions to test
    vector<int> klist = {1, 3, 5, 9, 15};                    // k values for k-NN
    int k_default = 5;    // k used while sweeping d
    int d_default = 40;   // d used while sweeping k

    // every combination to test, each one collects its own pooled confusion matrix
    struct Run { int LDAPCA, classifier, d, k; long conf[3][3] = {}; };
    vector<Run> runs;
    for (int d : dlist)
        for (int lp = 0; lp < 2; lp++)
            for (int cl = 0; cl < 2; cl++)
                if (!(lp == 1 && d < 2)) runs.push_back({lp, cl, d, k_default});   // LDA needs at least 2 PCA dims
    for (int kk : klist)
        for (int lp = 0; lp < 2; lp++)
            if (kk != k_default) runs.push_back({lp, 1, d_default, kk});
    int dmax = *max_element(dlist.begin(), dlist.end());

    for (int f = 0; f < n_buckets; f++) {
        vector<int> train_b, test_b = {f};
        for (int b = 0; b < n_buckets; b++) if (b != f) train_b.push_back(b);

        vector<Patch> train_data = combinebuckets(buckets, train_b);
        vector<Patch> test_data  = combinebuckets(buckets, test_b);

        // add a left-right mirrored copy of every training patch
        int n_orig = train_data.size();
        train_data.reserve(2 * n_orig);
        for (int i = 0; i < n_orig; i++) {
            Patch m = train_data[i];   // copies label, image, group too
            m.pixels = mirrorpatch(train_data[i].pixels, size, margin, true);
            train_data.push_back(m);
        }

        // PCA solved once per fold, with the most dimensions any run needs
        Eigenstuff eigall = eigenfinder(train_data, size, dmax, margin);

        // test patches z-scored and centred with the TRAINING mean, once per fold
        MatrixXf Xtest(size * size, test_data.size());
        for (int i = 0; i < (int)test_data.size(); i++) {
            Xtest.col(i) = Map<const Matrix<unsigned char, Dynamic, 1>>(test_data[i].pixels.data(), size*size).cast<float>();
            Xtest.col(i) = zscore(Xtest.col(i), size, margin);
        }
        Xtest.colwise() -= eigall.mean;

        for (Run &run : runs) {
            // keep only the top d directions: stored smallest first, so the last d
            int d = min(run.d, (int)eigall.eigvects.cols());
            Eigenstuff eigkept;
            eigkept.mean     = eigall.mean;
            eigkept.eigvals  = eigall.eigvals.tail(d);
            eigkept.eigvects = eigall.eigvects.rightCols(d);
            eigkept.Y        = eigall.Y.bottomRows(d);

            Eigenstuff eiglda;
            vector<Eigenstuff> classstats;
            if (run.LDAPCA == 1) {
                eiglda = LDA(eigkept, train_data);
                classstats = mdistclass(eiglda, train_data);
            } else {
                classstats = mdistclass(eigkept, train_data);
                classstats = adjeigvals(classstats, choice_eigvals, notsmall);
            }

            MatrixXf Ytest = eigkept.eigvects.transpose() * Xtest;   // d x ntest
            MatrixXf Ztest = (run.LDAPCA == 1) ? MatrixXf(eiglda.eigvects.transpose() * Ytest) : Ytest;
            const MatrixXf &Ztrain = (run.LDAPCA == 1) ? eiglda.Y : eigkept.Y;   // same space as Ztest

            float bias[3];
            for (int c = 0; c < 3; c++) bias[c] = -0.5f * classstats[c].eigvals.array().log().sum();

            for (int npatch = 0; npatch < Ztest.cols(); npatch++) {
                int maxim;
                if (run.classifier == 1) {
                    maxim = nnclass(Ztest.col(npatch), Ztrain, train_data, run.k);
                } else {
                    maxim = 1;
                    double scoreone = mahanobianscore(classstats[1], Ztest.col(npatch), bias[1]);
                    double scoretwo = mahanobianscore(classstats[2], Ztest.col(npatch), bias[2]);
                    if (scoretwo > scoreone) maxim = 2;
                    double scorezero = mahanobianscore(classstats[0], Ztest.col(npatch), bias[0]);
                    if (scorezero > scoreone && scorezero > scoretwo) maxim = 0;
                }
                run.conf[test_data[npatch].label][maxim]++;
            }
        }
        cout << "fold " << f << " done" << endl;
    }

    // pooled results: one line per run, in the terminal and in results.csv
    ofstream csv("results.csv");
    csv << "LDAPCA,classifier,d,k,BA,recall_clutter,recall_person,recall_cluster" << endl;
    for (const Run &run : runs) {
        double rec[3], ba = 0;
        for (int c = 0; c < 3; c++) {
            rec[c] = (double)run.conf[c][c] / (run.conf[c][0] + run.conf[c][1] + run.conf[c][2]);
            ba += rec[c];
        }
        ba = 100 * ba / 3;
        csv << run.LDAPCA << "," << run.classifier << "," << run.d << "," << run.k << "," << ba
            << "," << 100 * rec[0] << "," << 100 * rec[1] << "," << 100 * rec[2] << endl;
        cout << (run.LDAPCA ? "PCA+LDA " : "PCA     ") << (run.classifier ? "kNN  " : "Mahal ")
             << "d=" << run.d << " k=" << run.k << "  BA " << ba << endl;
    }
    return 0;
}
