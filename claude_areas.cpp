// Analysis by Claude: how many pixels are blobs of 1, 2, 3, 4+ persons, and what maxarea
// throws away big junk without throwing away people?
// A person belongs to the blob covering most of their box (label map).
// Same settings as preproc.cpp (1.52 x frame mean, r = 6, blobs >= 4 px).
// Uses claude_functions.cpp; preproc.cpp / functions.cpp are untouched.
//
// build (from albwork\):  g++ -std=c++17 -O2 claude_areas.cpp claude_functions.cpp -o claude_areas.exe
// run   (from albwork\):  .\claude_areas.exe

#include <stdio.h>
#include <iostream>
#include <vector>
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

const float BRIGHT = 1.52f;
const int MINAREA = 4;
const int R = 6;
const vector<int> ANGLES = {70, 80, 90};
const vector<int> MAXAREAS = {200, 300, 400, 500, 600, 800, 1000, 1500};

struct Box { int xmin, ymin, xmax, ymax; };
struct BlobRec { int alt, persons, area; };

int tagInt(const string &s, const string &tag, size_t from) {
    size_t a = s.find("<" + tag + ">", from) + tag.size() + 2;
    return stoi(s.substr(a, s.find("</" + tag + ">", a) - a));
}

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

int field(const string &name, int i) {
    stringstream ss(name);
    string t;
    for (int k = 0; k <= i; k++) getline(ss, t, '_');
    return stoi(t);
}

// mean, median, 90th percentile, max of a list of areas
void describe(vector<int> v, int &n, double &mean, int &med, int &p90, int &mx) {
    n = v.size();
    mean = 0; med = p90 = mx = 0;
    if (v.empty()) return;
    sort(v.begin(), v.end());
    for (int a : v) mean += a;
    mean /= n;
    med = v[n / 2];
    p90 = v[min(n - 1, (int)(0.9 * n))];
    mx = v.back();
}

int main(void) {
    vector<string> frames;
    for (auto &e : fs::directory_iterator(XML_DIR)) {
        string stem = e.path().stem().string();
        if (find(ANGLES.begin(), ANGLES.end(), field(stem, 2)) == ANGLES.end()) continue;
        if (personBoxes(XML_DIR + stem + ".xml").empty()) continue;
        frames.push_back(stem);
    }
    sort(frames.begin(), frames.end());

    vector<BlobRec> recs;
    int persons = 0;
    for (const string &stem : frames) {
        vector<Box> boxes = personBoxes(XML_DIR + stem + ".xml");
        int w, h, channels;
        unsigned char *data = stbi_load((IMG_DIR + stem + ".jpg").c_str(), &w, &h, &channels, 1);
        if (!data) { cout << "no image was loaded: " << stem << endl; continue; }
        for (Box &b : boxes) {
            b.xmin = max(0, b.xmin); b.ymin = max(0, b.ymin);
            b.xmax = min(w - 1, b.xmax); b.ymax = min(h - 1, b.ymax);
        }
        persons += boxes.size();
        double sum = 0;
        for (int i = 0; i < w * h; i++) sum += data[i];
        float mean = sum / (w * h);
        vector<char> mask(w * h);
        for (int i = 0; i < w * h; i++) mask[i] = claude_hotpixmask((float)data[i], mean, BRIGHT);
        stbi_image_free(data);

        vector<int> label;
        vector<Blob> blobs = claude_findblob((bool *)mask.data(), w, h, R, MINAREA, &label);
        vector<int> count(blobs.size(), 0);
        for (const Box &b : boxes) {
            map<int, int> cnt;
            for (int y = b.ymin; y <= b.ymax; y++)
                for (int x = b.xmin; x <= b.xmax; x++)
                    if (label[y * w + x] >= 0) cnt[label[y * w + x]]++;
            int best = 0, id = -1;
            for (auto &c : cnt) if (c.second > best) { best = c.second; id = c.first; }
            if (id >= 0) count[id]++;
        }
        int alt = field(stem, 1);
        for (size_t j = 0; j < blobs.size(); j++) recs.push_back({alt, count[j], blobs[j].area});
    }
    int nf = frames.size();

    printf("r = %d, threshold %.2f x frame mean, blobs >= %d px, %d frames, %d persons\n\n", R, BRIGHT, MINAREA, nf, persons);

    // ---- 1) blob area by number of persons, all altitudes and per altitude ----
    const char *groupName[] = {"junk (0)", "1 person", "2 persons", "3 persons", "4+ persons"};
    auto group = [](int n) { return min(n, 4); };
    printf("1) blob area in pixels, all altitudes\n");
    printf("   %-11s %6s %8s %8s %8s %8s\n", "blob", "count", "mean", "median", "90%", "max");
    for (int g = 0; g <= 4; g++) {
        vector<int> v;
        for (const BlobRec &b : recs) if (group(b.persons) == g) v.push_back(b.area);
        int n, med, p90, mx; double mean;
        describe(v, n, mean, med, p90, mx);
        printf("   %-11s %6d %8.0f %8d %8d %8d\n", groupName[g], n, mean, med, p90, mx);
    }

    map<int, int> framesAt;
    for (const string &s : frames) framesAt[field(s, 1)]++;
    printf("\n   median area per altitude (count in brackets)\n");
    printf("   %-6s %14s %14s %14s %14s\n", "alt", "1 person", "2 persons", "3 persons", "4+ persons");
    for (auto &fa : framesAt) {
        printf("   %-6d", fa.first);
        for (int g = 1; g <= 4; g++) {
            vector<int> v;
            for (const BlobRec &b : recs) if (b.alt == fa.first && group(b.persons) == g) v.push_back(b.area);
            int n, med, p90, mx; double mean;
            describe(v, n, mean, med, p90, mx);
            if (n == 0) printf(" %14s", "-");
            else printf(" %8d (%3d)", med, n);
        }
        printf("\n");
    }

    // ---- 2) what a single maxarea removes ----
    printf("\n2) blobs larger than maxarea thrown away (all altitudes)\n");
    printf("   %-8s %12s %14s %14s %14s %14s\n", "maxarea", "junk/frame", "junk removed", "persons lost", "2p blobs lost", "3+p blobs lost");
    int junkTotal = 0;
    for (const BlobRec &b : recs) if (b.persons == 0) junkTotal++;
    for (int m : MAXAREAS) {
        int junkLeft = 0, junkGone = 0, lost = 0, lost2 = 0, lost3 = 0;
        for (const BlobRec &b : recs) {
            bool gone = b.area > m;
            if (b.persons == 0) { if (gone) junkGone++; else junkLeft++; }
            if (gone) { lost += b.persons; if (b.persons == 2) lost2++; if (b.persons >= 3) lost3++; }
        }
        printf("   %-8d %12.1f %13.1f%% %8d (%4.1f%%) %14d %14d\n", m, (double)junkLeft / nf, 100.0 * junkGone / junkTotal,
               lost, 100.0 * lost / persons, lost2, lost3);
    }
    printf("   (without maxarea: %.1f junk blobs per frame)\n", (double)junkTotal / nf);
    return 0;
}
