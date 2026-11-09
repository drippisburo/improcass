// Analysis by Claude: what does the blob radius r in findblob change?
//   - persons found / split into several blobs
//   - extra (non-person) blobs
//   - how often two nearby persons end up in ONE blob (-> cluster), by how far apart their boxes are
// Same threshold and minimum area as functions.cpp (1.52 x frame mean, blobs >= 4 px).
// Uses copies of the functions (claude_functions.cpp); preproc.cpp / functions.cpp are untouched.
//
// build (from albwork\):  g++ -std=c++17 -O2 claude_radius.cpp claude_functions.cpp -o claude_radius.exe
// run   (from albwork\):  .\claude_radius.exe

#include <stdio.h>
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <map>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "claude_functions.h"

using namespace std;
namespace fs = std::filesystem;

const string IMG_DIR = "../split_daynight/night/JPEGImages/";
const string XML_DIR = "../split_daynight/night/Annotations/";

const float BRIGHT = 1.52f;                 // same as bright_threshold in functions.cpp
const int MINAREA = 4;                      // same as minarea in functions.cpp
const vector<int> RADII = {1, 2, 3, 4, 5, 6};
const vector<int> ANGLES = {70, 80, 90};
const int TOUCH_GAP = 1;                    // boxes overlapping or <= 1 px apart = touching (context.md)
const vector<int> GAP_EDGES = {1, 3, 6, 10};   // gap buckets: 0-1, 2-3, 4-6, 7-10, >10 px

struct Box { int xmin, ymin, xmax, ymax; };

// integer between <tag> and </tag>, searching from position `from`
int tagInt(const string &s, const string &tag, size_t from) {
    size_t a = s.find("<" + tag + ">", from) + tag.size() + 2;
    return stoi(s.substr(a, s.find("</" + tag + ">", a) - a));
}

// all Person boxes in one Pascal VOC XML file
vector<Box> personBoxes(const string &xmlPath) {
    ifstream in(xmlPath);
    stringstream ss;
    ss << in.rdbuf();
    string s = ss.str();
    vector<Box> boxes;
    size_t pos = 0;
    while ((pos = s.find("<object>", pos)) != string::npos) {
        size_t n = s.find("<name>", pos) + 6;
        if (s.substr(n, s.find("</name>", n) - n) == "Person")
            boxes.push_back({tagInt(s, "xmin", pos), tagInt(s, "ymin", pos),
                             tagInt(s, "xmax", pos), tagInt(s, "ymax", pos)});
        pos = s.find("</object>", pos);
    }
    return boxes;
}

// field i (0-based) of daynight_altitude_angle_0_frame
int field(const string &name, int i) {
    stringstream ss(name);
    string t;
    for (int k = 0; k <= i; k++) getline(ss, t, '_');
    return stoi(t);
}

// empty pixels between two boxes (0 if they touch or overlap), largest of the x and y gap
int gapBetween(const Box &a, const Box &b) {
    int gx = max(0, max(a.xmin, b.xmin) - min(a.xmax, b.xmax) - 1);
    int gy = max(0, max(a.ymin, b.ymin) - min(a.ymax, b.ymax) - 1);
    return max(gx, gy);
}

int gapBucket(int gap) {
    for (size_t k = 0; k < GAP_EDGES.size(); k++) if (gap <= GAP_EDGES[k]) return k;
    return GAP_EDGES.size();
}

bool inside(const Box &b, double x, double y) {
    return x >= b.xmin && x <= b.xmax && y >= b.ymin && y <= b.ymax;
}

struct RStats {
    int found = 0;          // >= 1 blob centred in the person box
    int covered = 0;        // >= 1 kept-blob pixel in the person box
    int split = 0;          // >= 2 blobs centred in the person box
    int blobs = 0, extra = 0;
    vector<int> merged = vector<int>(GAP_EDGES.size() + 1, 0);   // pairs sharing one blob, per gap bucket
    int clusterBlobs = 0, clusterTouching = 0, clusterApartSmall = 0, clusterApartBig = 0;
    map<int, int> splitByAlt;
};

int main(void) {
    // all frames at the chosen angles that contain at least one person
    vector<string> frames;
    for (auto &e : fs::directory_iterator(XML_DIR)) {
        string stem = e.path().stem().string();
        if (find(ANGLES.begin(), ANGLES.end(), field(stem, 2)) == ANGLES.end()) continue;
        if (personBoxes(XML_DIR + stem + ".xml").empty()) continue;
        frames.push_back(stem);
    }
    sort(frames.begin(), frames.end());

    vector<RStats> st(RADII.size());
    vector<int> pairsPerBucket(GAP_EDGES.size() + 1, 0);
    map<int, int> personsByAlt;
    int persons = 0;
    vector<string> splitFixed, pairJoined;   // examples: split at r=1 but one blob at r=2, touching pair joined only at r=2

    for (const string &stem : frames) {
        vector<Box> boxes = personBoxes(XML_DIR + stem + ".xml");
        int w, h, channels;
        unsigned char *data = stbi_load((IMG_DIR + stem + ".jpg").c_str(), &w, &h, &channels, 1);
        if (!data) { cout << "no image was loaded: " << stem << endl; continue; }
        for (Box &b : boxes) {
            b.xmin = max(0, b.xmin); b.ymin = max(0, b.ymin);
            b.xmax = min(w - 1, b.xmax); b.ymax = min(h - 1, b.ymax);
        }
        int nP = boxes.size(), alt = field(stem, 1);
        persons += nP;
        personsByAlt[alt] += nP;

        // ground truth: every pair of persons in the frame and how far apart their boxes are
        vector<array<int, 3>> pairs;   // i, j, gap
        for (int i = 0; i < nP; i++)
            for (int j = i + 1; j < nP; j++) {
                int g = gapBetween(boxes[i], boxes[j]);
                pairs.push_back({i, j, g});
                pairsPerBucket[gapBucket(g)]++;
            }

        double sum = 0;
        for (int i = 0; i < w * h; i++) sum += data[i];
        float mean = sum / (w * h);
        vector<char> mask(w * h);
        for (int i = 0; i < w * h; i++) mask[i] = claude_hotpixmask((float)data[i], mean, BRIGHT);

        vector<vector<int>> centred(RADII.size(), vector<int>(nP, 0));   // blobs centred in each person box
        vector<vector<int>> dom(RADII.size(), vector<int>(nP, -1));      // blob covering most of each person box

        for (size_t k = 0; k < RADII.size(); k++) {
            vector<char> work = mask;   // findblob switches pixels off, so give it a copy
            vector<int> label;
            vector<Blob> blobs = claude_findblob((bool *)work.data(), w, h, RADII[k], MINAREA, &label);
            RStats &s = st[k];
            s.blobs += blobs.size();

            for (int p = 0; p < nP; p++) {
                const Box &b = boxes[p];
                for (const Blob &bl : blobs) if (inside(b, bl.mx, bl.my)) centred[k][p]++;
                map<int, int> count;
                for (int y = b.ymin; y <= b.ymax; y++)
                    for (int x = b.xmin; x <= b.xmax; x++)
                        if (label[y * w + x] >= 0) count[label[y * w + x]]++;
                int best = 0;
                for (auto &c : count) if (c.second > best) { best = c.second; dom[k][p] = c.first; }
                if (centred[k][p] >= 1) s.found++;
                if (centred[k][p] >= 2) { s.split++; s.splitByAlt[alt]++; }
                if (dom[k][p] >= 0) s.covered++;
            }

            for (const Blob &bl : blobs) {
                // which persons have their box centre inside this blob's box (the labelling rule in context.md)
                vector<int> in;
                bool centredOnPerson = false;
                for (int p = 0; p < nP; p++) {
                    const Box &b = boxes[p];
                    if (inside(b, bl.mx, bl.my)) centredOnPerson = true;
                    double cx = (b.xmin + b.xmax) / 2.0, cy = (b.ymin + b.ymax) / 2.0;
                    if (cx >= bl.xmin && cx <= bl.xmax && cy >= bl.ymin && cy <= bl.ymax) in.push_back(p);
                }
                if (!centredOnPerson) s.extra++;
                if (in.size() >= 2) {
                    s.clusterBlobs++;
                    bool touching = false;
                    for (size_t a = 0; a < in.size() && !touching; a++)
                        for (size_t c = a + 1; c < in.size(); c++)
                            if (gapBetween(boxes[in[a]], boxes[in[c]]) <= TOUCH_GAP) { touching = true; break; }
                    if (touching) s.clusterTouching++;
                    else if (bl.area > 1000) s.clusterApartBig++;
                    else s.clusterApartSmall++;
                }
            }

            for (auto &pr : pairs) {
                int a = pr[0], c = pr[1];
                if (dom[k][a] >= 0 && dom[k][a] == dom[k][c]) s.merged[gapBucket(pr[2])]++;
            }
        }

        // examples (r=1 is index 0, r=2 is index 1)
        for (int p = 0; p < nP; p++)
            if (centred[0][p] >= 2 && centred[1][p] == 1 && splitFixed.size() < 6) {
                const Box &b = boxes[p];
                splitFixed.push_back(stem + "  box (" + to_string(b.xmin) + "," + to_string(b.ymin) + ")-(" +
                                     to_string(b.xmax) + "," + to_string(b.ymax) + ")");
            }
        for (auto &pr : pairs) {
            int a = pr[0], c = pr[1];
            bool sep1 = !(dom[0][a] >= 0 && dom[0][a] == dom[0][c]);
            bool join2 = dom[1][a] >= 0 && dom[1][a] == dom[1][c];
            if (pr[2] <= TOUCH_GAP && sep1 && join2 && pairJoined.size() < 6)
                pairJoined.push_back(stem + "  boxes (" + to_string(boxes[a].xmin) + "," + to_string(boxes[a].ymin) +
                                     ") and (" + to_string(boxes[c].xmin) + "," + to_string(boxes[c].ymin) + ")");
        }
        stbi_image_free(data);
    }

    int nf = frames.size();
    printf("threshold %.2f x frame mean, blobs >= %d px, %d frames, %d persons (angles", BRIGHT, MINAREA, nf, persons);
    for (int a : ANGLES) printf(" %d", a);
    printf(")\n\n");

    printf("1) detection and splitting\n");
    printf("%-4s %16s %16s %14s %12s %12s\n", "r", "found (centred)", "covered", "split (>=2)", "blobs/frame", "extra/frame");
    for (size_t k = 0; k < RADII.size(); k++) {
        RStats &s = st[k];
        printf("%-4d %7.2f%% (%4d) %7.2f%% (%4d) %6.2f%% (%3d) %12.1f %12.1f\n", RADII[k],
               100.0 * s.found / persons, persons - s.found, 100.0 * s.covered / persons, persons - s.covered,
               100.0 * s.split / persons, s.split, (double)s.blobs / nf, (double)s.extra / nf);
    }
    printf("(found/covered: numbers in brackets = persons missed)\n");

    printf("\n   split persons by altitude\n%-4s", "r");
    for (auto &a : personsByAlt) printf(" %7dm", a.first);
    printf("\n");
    for (size_t k = 0; k < RADII.size(); k++) {
        printf("%-4d", RADII[k]);
        for (auto &a : personsByAlt) printf(" %7.1f%%", 100.0 * st[k].splitByAlt[a.first] / a.second);
        printf("\n");
    }

    printf("\n2) pairs of persons that end up in ONE blob, by gap between their boxes\n");
    const char *names[] = {"0-1 px (touching)", "2-3 px", "4-6 px", "7-10 px", ">10 px"};
    printf("%-20s %7s", "gap", "pairs");
    for (int r : RADII) printf("     r=%d", r);
    printf("\n");
    for (size_t g = 0; g <= GAP_EDGES.size(); g++) {
        printf("%-20s %7d", names[g], pairsPerBucket[g]);
        for (size_t k = 0; k < RADII.size(); k++)
            printf(" %7.1f%%", pairsPerBucket[g] ? 100.0 * st[k].merged[g] / pairsPerBucket[g] : 0.0);
        printf("\n");
    }

    printf("\n3) blobs that would be labelled 'cluster' (>= 2 person-box centres inside the blob's box)\n");
    printf("%-4s %10s %20s %22s %22s\n", "r", "clusters", "persons touching", "persons apart, small", "apart, blob > 1000 px");
    for (size_t k = 0; k < RADII.size(); k++) {
        RStats &s = st[k];
        printf("%-4d %10d %20d %22d %22d\n", RADII[k], s.clusterBlobs, s.clusterTouching, s.clusterApartSmall, s.clusterApartBig);
    }

    printf("\nexamples: person split in 2+ blobs at r=1, one blob at r=2\n");
    for (const string &e : splitFixed) printf("   %s\n", e.c_str());
    printf("examples: touching pair separate at r=1, one blob at r=2\n");
    for (const string &e : pairJoined) printf("   %s\n", e.c_str());
    return 0;
}
