// Analysis by Claude: increase the blob radius r until merging becomes harmful.
// Goal (yours): people close together SHOULD end up in one blob (-> cluster class).
// So per r we count good and bad effects separately:
//   good: touching pairs (boxes <= 1 px apart) in one blob, number of multi-person blobs
//   bad:  pairs far apart in one blob (chaining), persons swallowed by warm junk,
//         multi-person blobs too wide for a 38 px patch, persons not covered by any blob
// A person belongs to the blob covering most of their box (label map).
// Same threshold and minimum area as functions.cpp (1.52 x frame mean, blobs >= 4 px).
// Uses claude_functions.cpp; preproc.cpp / functions.cpp are untouched.
//
// build (from albwork\):  g++ -std=c++17 -O2 claude_rsweep.cpp claude_functions.cpp -o claude_rsweep.exe
// run   (from albwork\):  .\claude_rsweep.exe

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

const float BRIGHT = 1.52f;
const int MINAREA = 4;
const int R_MAX = 12;
const vector<int> ANGLES = {70, 80, 90};
const int PATCH = 38;                       // planned patch size at 60 m (context.md)

struct Box { int xmin, ymin, xmax, ymax; };

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

int gapBetween(const Box &a, const Box &b) {
    int gx = max(0, max(a.xmin, b.xmin) - min(a.xmax, b.xmax) - 1);
    int gy = max(0, max(a.ymin, b.ymin) - min(a.ymax, b.ymax) - 1);
    return max(gx, gy);
}

// altitude bands: 60 m has most persons and big people, 70-90 m, 100-130 m small people
int band(int alt) { return alt <= 60 ? 0 : alt <= 90 ? 1 : 2; }
const char *BAND_NAME[] = {"60m", "70-90m", "100-130m"};

// pair gap groups
int gapGroup(int g) { return g <= 1 ? 0 : g <= 6 ? 1 : g <= 15 ? 2 : 3; }
const char *GAP_NAME[] = {"0-1px", "2-6px", "7-15px", ">15px"};

struct RStats {
    int covered = 0, swallowed = 0, junkBlobs = 0;
    int multi = 0, multi2 = 0, multi34 = 0, multi5 = 0, multiWide = 0, personsInMulti = 0;
    array<array<int, 4>, 3> merged{};   // [band][gap group] pairs in one blob
};

int main(void) {
    vector<string> frames;
    for (auto &e : fs::directory_iterator(XML_DIR)) {
        string stem = e.path().stem().string();
        if (find(ANGLES.begin(), ANGLES.end(), field(stem, 2)) == ANGLES.end()) continue;
        if (personBoxes(XML_DIR + stem + ".xml").empty()) continue;
        frames.push_back(stem);
    }
    sort(frames.begin(), frames.end());

    vector<RStats> st(R_MAX + 1);
    array<array<int, 4>, 3> pairs{};
    array<int, 3> personsInBand{};
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
        int nP = boxes.size(), bd = band(field(stem, 1));
        persons += nP;
        personsInBand[bd] += nP;

        vector<array<int, 3>> pr;   // i, j, gap group
        for (int i = 0; i < nP; i++)
            for (int j = i + 1; j < nP; j++) {
                int g = gapGroup(gapBetween(boxes[i], boxes[j]));
                pr.push_back({i, j, g});
                pairs[bd][g]++;
            }

        vector<char> inAnyBox(w * h, 0);
        for (const Box &b : boxes)
            for (int y = b.ymin; y <= b.ymax; y++)
                for (int x = b.xmin; x <= b.xmax; x++) inAnyBox[y * w + x] = 1;

        double sum = 0;
        for (int i = 0; i < w * h; i++) sum += data[i];
        float mean = sum / (w * h);
        vector<char> mask(w * h);
        for (int i = 0; i < w * h; i++) mask[i] = claude_hotpixmask((float)data[i], mean, BRIGHT);

        for (int r = 1; r <= R_MAX; r++) {
            vector<char> work = mask;
            vector<int> label;
            vector<Blob> blobs = claude_findblob((bool *)work.data(), w, h, r, MINAREA, &label);
            RStats &s = st[r];

            vector<int> boxPx(blobs.size(), 0);   // pixels of each blob inside some person box
            for (int i = 0; i < w * h; i++) if (label[i] >= 0 && inAnyBox[i]) boxPx[label[i]]++;

            vector<int> owner(nP, -1), owners(blobs.size(), 0);
            for (int p = 0; p < nP; p++) {
                const Box &b = boxes[p];
                map<int, int> cnt;
                for (int y = b.ymin; y <= b.ymax; y++)
                    for (int x = b.xmin; x <= b.xmax; x++)
                        if (label[y * w + x] >= 0) cnt[label[y * w + x]]++;
                int best = 0;
                for (auto &c : cnt) if (c.second > best) { best = c.second; owner[p] = c.first; }
                if (owner[p] < 0) continue;
                s.covered++;
                owners[owner[p]]++;
                // swallowed: the person's blob is mostly pixels outside every person box
                if (blobs[owner[p]].area - boxPx[owner[p]] > blobs[owner[p]].area / 2) s.swallowed++;
            }

            for (size_t j = 0; j < blobs.size(); j++) {
                int n = owners[j];
                if (n == 0) { s.junkBlobs++; continue; }
                if (n == 1) continue;
                s.multi++;
                s.personsInMulti += n;
                if (n == 2) s.multi2++; else if (n <= 4) s.multi34++; else s.multi5++;
                if (max(blobs[j].xmax - blobs[j].xmin + 1, blobs[j].ymax - blobs[j].ymin + 1) > PATCH) s.multiWide++;
            }

            for (auto &q : pr)
                if (owner[q[0]] >= 0 && owner[q[0]] == owner[q[1]]) s.merged[bd][q[2]]++;
        }
        stbi_image_free(data);
    }

    int nf = frames.size();
    printf("threshold %.2f x frame mean, blobs >= %d px, %d frames, %d persons\n\n", BRIGHT, MINAREA, nf, persons);

    printf("A) all altitudes\n");
    printf("%-3s %8s %10s %10s %7s %7s %5s %10s %10s %10s %12s %12s\n", "r", "covered", "swallowed", "multi", "2 ppl", "3-4", "5+",
           "wide>38px", "touching", "7-15px", ">15px", "junk/frame");
    for (int r = 1; r <= R_MAX; r++) {
        RStats &s = st[r];
        int tm = 0, tp = 0, fm = 0, ffm = 0, fp = 0;
        for (int b = 0; b < 3; b++) {
            tm += s.merged[b][0]; tp += pairs[b][0];
            fm += s.merged[b][2]; fp += pairs[b][2];
            ffm += s.merged[b][3];
        }
        printf("%-3d %7.2f%% %10d %10d %7d %7d %5d %10d %9.1f%% %9.1f%% %12d %12.1f\n", r, 100.0 * s.covered / persons, s.swallowed,
               s.multi, s.multi2, s.multi34, s.multi5, s.multiWide, 100.0 * tm / tp, 100.0 * fm / fp, ffm, (double)s.junkBlobs / nf);
    }
    printf("(covered: person part of a blob; swallowed: person's blob mostly outside all person boxes;\n");
    printf(" multi: blobs holding >= 2 persons; touching / 7-15px: share of such pairs in one blob; >15px: number of such pairs in one blob)\n");

    for (int b = 0; b < 3; b++) {
        printf("\nB) %s (%d persons): share of pairs in one blob, by gap between boxes\n", BAND_NAME[b], personsInBand[b]);
        printf("%-3s", "r");
        for (int g = 0; g < 4; g++) printf(" %8s(%5d)", GAP_NAME[g], pairs[b][g]);
        printf("\n");
        for (int r = 1; r <= R_MAX; r++) {
            printf("%-3d", r);
            for (int g = 0; g < 4; g++) printf(" %14.1f%%", pairs[b][g] ? 100.0 * st[r].merged[b][g] / pairs[b][g] : 0.0);
            printf("\n");
        }
    }
    return 0;
}
