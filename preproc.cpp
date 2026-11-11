#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <string>
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

    long pooled[3][3] = {};
    for (int f = 0; f < n_buckets; f++) {
        vector<int> train_b, test_b = {f};
        for (int b = 0; b < n_buckets; b++) if (b != f) train_b.push_back(b);

        vector<Patch> train_data = combinebuckets(buckets, train_b);
        vector<Patch> test_data  = combinebuckets(buckets, test_b);

        // your existing code, moved in unchanged:
        // add a left-right mirrored copy of every training patch
        int n_orig = train_data.size();
        train_data.reserve(2 * n_orig);
        for (int i = 0; i < n_orig; i++) {
            Patch m = train_data[i];   // copies label, image, group too
            m.pixels = mirrorpatch(train_data[i].pixels, size, margin, true);
            train_data.push_back(m);
        }



        // PCA applied, kept dimensions = kept, all data stored in struct
        Eigenstuff eigkept = eigenfinder( train_data , size, kept, margin);

        //visualization stuff ACTIVATE FOR VISUALIZATION 
        
        int n_show = min(3, (int)eigkept.eigvects.cols());
        for (int n = 0; n < n_show; n++) {
            VectorXf v = eigkept.eigvects.col(eigkept.eigvects.cols() - 1 - n).cwiseAbs();
            float hi = v.maxCoeff();

            vector<unsigned char> img(v.size());
            for (int i = 0; i < v.size(); i++) {
                img[i] = (unsigned char)(v(i) / hi * 255.0f);
            }
            makeimage(img.data(), size, size, ("eigenvector_top_" + to_string(f) + "_" + to_string(n + 1) + ".png").c_str());
            
            // int K = eigkept.eigvals.size();
            // cout << "eigenvalue 1: " << eigkept.eigvals(K - 1) << endl;
            // cout << "eigenvalue 2: " << eigkept.eigvals(K - 2) << endl;
            // cout << "ratio: " << eigkept.eigvals(K - 1) / eigkept.eigvals(K - 2) << endl;
            
        }
        
        //now onto LDA  ------ if it is activated
        //
        Eigenstuff eiglda;               
        vector<Eigenstuff> classstats;

        if (LDAPCA == 1) {
            eiglda = LDA(eigkept, train_data);

            // LDA directions back to pixel space so they can be seen as masks
            //       ----visualization----
            /*
            int n_lda = min(2, (int)eiglda.eigvects.cols());
            for (int n = 0; n < n_lda; n++) {
                VectorXf v = eigkept.eigvects * eiglda.eigvects.col(eiglda.eigvects.cols() - 1 - n);
                float hi = v.cwiseAbs().maxCoeff();

                vector<unsigned char> img(v.size());
                for (int i = 0; i < v.size(); i++) {
                    img[i] = (unsigned char)(127.5f + 127.5f * v(i) / hi);   // grey = 0, white = +, black = -
                }
                makeimage(img.data(), size, size, ("lda_vector_" + to_string(f) + to_string(n + 1) + ".png").c_str());
                // cout << "LDA eigenvalue " << n + 1 << ": " << eiglda.eigvals(eiglda.eigvals.size() - 1 - n) << endl;
            }
            */
           classstats = mdistclass(eiglda, train_data);

        }
        // mean, covariance eigenvalues and eigenvectors per class, index = label

        // choice_eigvals: 0 is use largest of small ones, 1 is use average eigval
        // this basically only applies if you dont do LDA otherwise only 2 dimensions are kept and this doesnt make much sense
        
        if (LDAPCA == 0){ //only PCA
            classstats = mdistclass(eigkept, train_data);
            classstats = adjeigvals(classstats, choice_eigvals, notsmall);  // to turn off if LDA is applied 
        }
        // finally use the test data and project it into the 2 dimensions
        // using the trained data from PCA (and LDA)
        MatrixXf Xtest(size * size, test_data.size());
        for (int i = 0; i < (int)test_data.size(); i++) {
            Xtest.col(i) = Map<const Matrix<unsigned char, Dynamic, 1>>(test_data[i].pixels.data(), size*size).cast<float>();
            Xtest.col(i) = zscore(Xtest.col(i), size, margin);
        }
        Xtest.colwise() -= eigkept.mean;                         // TRAINING mean
        MatrixXf Ytest = eigkept.eigvects.transpose() * Xtest;  // kept x ntest
        
        MatrixXf Ztest;
        if (LDAPCA == 1) Ztest = eiglda.eigvects.transpose() * Ytest;   // 2 x ntest
        else Ztest = Ytest;

        const MatrixXf &Ztrain = (LDAPCA == 1) ? eiglda.Y : eigkept.Y;   // same space as Ztest

        int conf[3][3] = {};
        float bias[3];
        for (int c = 0; c < 3; c++) bias[c] = -0.5f * classstats[c].eigvals.array().log().sum();

        int maxim;
        for ( int npatch = 0; npatch < Ztest.cols(); npatch++){
            if (classifier == 1) {
                maxim = nnclass(Ztest.col(npatch), Ztrain, train_data, k);
            }
            else{
                maxim = 1;
                double scoreone = mahanobianscore(classstats[1], Ztest.col(npatch) , bias[1] );
                double scoretwo = mahanobianscore(classstats[2], Ztest.col(npatch) , bias[2] );
                if (scoretwo > scoreone){maxim = 2;}
                double scorezero = mahanobianscore(classstats[0], Ztest.col(npatch) , bias[0] );
                if (scorezero > scoreone && scorezero > scoretwo){maxim = 0;}
            }
            conf[test_data[npatch].label][maxim]++;
        }

        //finally the confusion matrix to assess the performance
        cout << "confusion matrix: rows = true label, columns = assigned label" << endl;
        for (int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++){
                cout << conf[i][j] << " ";
            }
            cout << endl;
        }
        
        //balance accuracy to see how well this combo predicts the remaining bucket
        double ba = 0;
        for (int c = 0; c < 3; c++) {
            ba += (double)conf[c][c] / (conf[c][0] + conf[c][1] + conf[c][2]);
            for (int j = 0; j < 3; j++) pooled[c][j] += conf[c][j];
        }
        cout << "fold " << f << ": BA " << 100 * ba / 3 << endl;
    }
    // then print pooled[][] and its BA, computed the same way
    cout << "pooled confusion matrix" << endl;
    double pba = 0;
    for (int c = 0; c < 3; c++) {
        for (int j = 0; j < 3; j++) cout << pooled[c][j] << " ";
        cout << endl;
        pba += (double)pooled[c][c] / (pooled[c][0] + pooled[c][1] + pooled[c][2]);
    }
    cout << "pooled BA " << 100 * pba / 3 << endl;

    
    return 0;
}