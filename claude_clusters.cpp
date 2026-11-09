// Analysis by Claude, blob radius r = 6:
//   1) how many blobs hold 3+ persons, and what shape they have (vs 1- and 2-person blobs)
//   2) a size limit per altitude = 3 x (median single-person blob area) x MARGIN,
//      and how many blobs are above it (overall, and on one frame per altitude / flight)
// A person belongs to the blob covering most of their box (label map).
// Same threshold and minimum area as functions.cpp (1.52 x frame mean, blobs >= 4 px).
// Uses claude_functions.cpp; preproc.cpp / functions.cpp are untouched.
//
// build (from albwork\):  g++ -std=c++17 -O2 claude_clusters.cpp claude_functions.cpp -o claude_clusters.exe
// run   (from albwork\):  .\claude_clusters.exe            (text only)
//                         .\claude_clusters.exe outdir     (also writes pictures into outdir)

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <random>
#include <map>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "claude_functions.h"

using namespace std;
namespace fs = std::filesystem;

const string IMG_DIR = "../split_daynight/night/JPEGImages/";
const string XML_DIR = "../split_daynight/night/Annotations/";

const float BRIGHT = 1.52f;
const int MINAREA = 4;
const int R = 6;
const double MARGIN = 1.5;      // limit = 3 people x median single-person area x MARGIN
const vector<int> ANGLES = {70, 80, 90};

struct Box { int xmin, ymin, xmax, ymax; };

// one blob with everything we want to know about it
struct BlobInfo {
    string frame;
    int alt, persons, area, longSide, shortSide;
    double elong, fill;
    Blob b;
};

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

double median(vector<double> v) {
    if (v.empty()) return 0;
    sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// everything one frame gives us: image, boxes, blobs, label map, persons per blob
struct Frame {
    int w = 0, h = 0, alt = 0;
    vector<unsigned char> img;
    vector<Box> boxes;
    vector<Blob> blobs;
    vector<int> label, owners;
    vector<BlobInfo> info;
};

bool loadFrame(const string &stem, Frame &F) {
    F.boxes = personBoxes(XML_DIR + stem + ".xml");
    int channels;
    unsigned char *data = stbi_load((IMG_DIR + stem + ".jpg").c_str(), &F.w, &F.h, &channels, 1);
    if (!data) return false;
    int w = F.w, h = F.h;
    F.img.assign(data, data + w * h);
    stbi_image_free(data);
    F.alt = field(stem, 1);
    for (Box &b : F.boxes) {
        b.xmin = max(0, b.xmin); b.ymin = max(0, b.ymin);
        b.xmax = min(w - 1, b.xmax); b.ymax = min(h - 1, b.ymax);
    }

    double sum = 0;
    for (int i = 0; i < w * h; i++) sum += F.img[i];
    float mean = sum / (w * h);
    vector<char> mask(w * h);
    for (int i = 0; i < w * h; i++) mask[i] = claude_hotpixmask((float)F.img[i], mean, BRIGHT);
    F.blobs = claude_findblob((bool *)mask.data(), w, h, R, MINAREA, &F.label);

    // persons per blob: each person goes to the blob covering most of their box
    F.owners.assign(F.blobs.size(), 0);
    for (const Box &b : F.boxes) {
        map<int, int> cnt;
        for (int y = b.ymin; y <= b.ymax; y++)
            for (int x = b.xmin; x <= b.xmax; x++)
                if (F.label[y * w + x] >= 0) cnt[F.label[y * w + x]]++;
        int best = 0, id = -1;
        for (auto &c : cnt) if (c.second > best) { best = c.second; id = c.first; }
        if (id >= 0) F.owners[id]++;
    }

    // shape: second moments of the blob pixels -> elongation (long axis / short axis)
    size_t nb = F.blobs.size();
    vector<double> sxx(nb, 0), syy(nb, 0), sxy(nb, 0);
    for (int i = 0; i < w * h; i++) {
        int l = F.label[i];
        if (l < 0) continue;
        double dx = i % w - F.blobs[l].mx, dy = i / w - F.blobs[l].my;
        sxx[l] += dx * dx; syy[l] += dy * dy; sxy[l] += dx * dy;
    }
    for (size_t j = 0; j < nb; j++) {
        const Blob &bl = F.blobs[j];
        double a = sxx[j] / bl.area, c = syy[j] / bl.area, b = sxy[j] / bl.area;
        double root = sqrt((a - c) * (a - c) / 4 + b * b);
        double l1 = (a + c) / 2 + root, l2 = max((a + c) / 2 - root, 0.25);
        int bw = bl.xmax - bl.xmin + 1, bh = bl.ymax - bl.ymin + 1;
        F.info.push_back({stem, F.alt, F.owners[j], bl.area, max(bw, bh), min(bw, bh),
                          sqrt(l1 / l2), (double)bl.area / (bw * bh), bl});
    }
    return true;
}

// grey image, blob pixels coloured, person boxes green; `crop` = region to draw, zoom = scale
void drawRegion(const Frame &F, int x0, int y0, int x1, int y1, int zoom, const vector<int> &colourOfBlob,
                vector<unsigned char> &out, int outW, int ox, int oy) {
    unsigned char col[4][3] = {{60, 140, 255}, {255, 210, 0}, {255, 40, 40}, {200, 0, 200}};
    int w = F.w;
    for (int y = y0; y <= y1; y++)
        for (int x = x0; x <= x1; x++) {
            if (x < 0 || y < 0 || x >= F.w || y >= F.h) continue;
            int i = y * w + x;
            unsigned char r = F.img[i], g = F.img[i], b = F.img[i];
            int l = F.label[i];
            if (l >= 0 && colourOfBlob[l] >= 0) { int k = colourOfBlob[l]; r = col[k][0]; g = col[k][1]; b = col[k][2]; }
            for (const Box &bx : F.boxes) {
                bool e = ((x == bx.xmin || x == bx.xmax) && y >= bx.ymin && y <= bx.ymax) ||
                         ((y == bx.ymin || y == bx.ymax) && x >= bx.xmin && x <= bx.xmax);
                if (e) { r = 0; g = 230; b = 0; }
            }
            for (int zy = 0; zy < zoom; zy++)
                for (int zx = 0; zx < zoom; zx++) {
                    int o = ((oy + (y - y0) * zoom + zy) * outW + ox + (x - x0) * zoom + zx) * 3;
                    out[o] = r; out[o + 1] = g; out[o + 2] = b;
                }
        }
}

int main(int argc, char **argv) {
    string outDir = argc > 1 ? argv[1] : "";

    vector<string> frames;
    for (auto &e : fs::directory_iterator(XML_DIR)) {
        string stem = e.path().stem().string();
        if (find(ANGLES.begin(), ANGLES.end(), field(stem, 2)) == ANGLES.end()) continue;
        if (personBoxes(XML_DIR + stem + ".xml").empty()) continue;
        frames.push_back(stem);
    }
    sort(frames.begin(), frames.end());

    // ---- pass 1: all blobs of all frames ----
    vector<BlobInfo> all;
    map<int, int> framesAtAlt;
    for (const string &stem : frames) {
        Frame F;
        if (!loadFrame(stem, F)) continue;
        framesAtAlt[F.alt]++;
        all.insert(all.end(), F.info.begin(), F.info.end());
    }

    printf("r = %d, threshold %.2f x frame mean, blobs >= %d px, %d frames\n\n", R, BRIGHT, MINAREA, (int)frames.size());

    // ---- 1) multi-person blobs and their shape ----
    map<int, int> byN;
    for (const BlobInfo &b : all) if (b.persons > 0) byN[min(b.persons, 6)]++;
    printf("1) blobs by number of persons inside: ");
    for (auto &p : byN) printf(" %d%s: %d ", p.first, p.first == 6 ? "+" : "", p.second);
    printf("\n\n   shape (medians)\n");
    printf("   %-12s %6s %8s %12s %12s %12s %8s\n", "persons", "blobs", "area", "long side", "short side", "elongation", "fill");
    const char *groupName[] = {"1", "2", "3+"};
    for (int g = 0; g < 3; g++) {
        vector<double> a, ls, ss, el, fi;
        for (const BlobInfo &b : all) {
            int grp = b.persons >= 3 ? 2 : b.persons - 1;
            if (b.persons == 0 || grp != g) continue;
            a.push_back(b.area); ls.push_back(b.longSide); ss.push_back(b.shortSide);
            el.push_back(b.elong); fi.push_back(b.fill);
        }
        printf("   %-12s %6d %8.0f %12.0f %12.0f %12.2f %8.2f\n", groupName[g], (int)a.size(), median(a), median(ls),
               median(ss), median(el), median(fi));
    }
    printf("   (elongation = long axis / short axis of the pixel spread; fill = blob area / its box area)\n");

    printf("\n   every blob with 3+ persons\n");
    printf("   %-20s %4s %7s %6s %9s %6s %6s\n", "frame", "ppl", "area", "box", "", "elong", "fill");
    vector<BlobInfo> big3;
    for (const BlobInfo &b : all) if (b.persons >= 3) big3.push_back(b);
    sort(big3.begin(), big3.end(), [](const BlobInfo &a, const BlobInfo &b) { return a.alt != b.alt ? a.alt < b.alt : a.persons > b.persons; });
    for (const BlobInfo &b : big3)
        printf("   %-20s %4d %7d %4dx%-4d @(%3d,%3d) %6.2f %6.2f\n", b.frame.c_str(), b.persons, b.area, b.longSide, b.shortSide,
               b.b.xmin, b.b.ymin, b.elong, b.fill);

    // ---- 2) size limit per altitude ----
    map<int, double> limitAt;
    printf("\n2) size limit per altitude = 3 x median single-person area x %.1f\n", MARGIN);
    printf("   %-5s %7s %10s %12s %10s %10s %12s %14s %12s\n", "alt", "frames", "1p median", "3+p areas", "limit", "blobs >", "per frame",
           "with persons", "persons in");
    for (auto &fa : framesAtAlt) {
        int alt = fa.first;
        vector<double> single;
        for (const BlobInfo &b : all) if (b.alt == alt && b.persons == 1) single.push_back(b.area);
        double limit = 3 * median(single) * MARGIN;
        limitAt[alt] = limit;
        int over = 0, overWithPersons = 0, personsInOver = 0;
        string threeAreas;
        for (const BlobInfo &b : all) {
            if (b.alt != alt) continue;
            if (b.persons >= 3) threeAreas += (threeAreas.empty() ? "" : ",") + to_string(b.area);
            if (b.area > limit) { over++; if (b.persons > 0) { overWithPersons++; personsInOver += b.persons; } }
        }
        if (threeAreas.empty()) threeAreas = "-";
        if (threeAreas.size() > 12) threeAreas = threeAreas.substr(0, 9) + "...";
        printf("   %-5d %7d %10.0f %12s %10.0f %10d %12.2f %14d %12d\n", alt, fa.second, median(single), threeAreas.c_str(), limit,
               over, (double)over / fa.second, overWithPersons, personsInOver);
    }
    printf("   (blobs >: blobs larger than the limit; with persons / persons in: how many of them hold persons, and how many)\n");
    int threeOver = 0;
    for (const BlobInfo &b : big3) if (b.area > limitAt[b.alt]) threeOver++;
    printf("   3+ person blobs above their own altitude limit: %d of %d\n", threeOver, (int)big3.size());

    // ---- 3) one random frame per altitude (different flights) ----
    printf("\n3) one random frame per altitude\n");
    mt19937 rng(7);
    vector<string> sample;
    for (auto &fa : framesAtAlt) {
        vector<string> atAlt;
        for (const string &s : frames) if (field(s, 1) == fa.first) atAlt.push_back(s);
        sample.push_back(atAlt[rng() % atAlt.size()]);
    }
    for (const string &stem : sample) {
        Frame F;
        if (!loadFrame(stem, F)) continue;
        double limit = limitAt[F.alt];
        int over = 0;
        string list;
        for (const BlobInfo &b : F.info)
            if (b.area > limit) {
                over++;
                list += "  " + to_string(b.area) + "px/" + to_string(b.persons) + "p";
            }
        printf("   %-20s persons %2d, blobs %3d, limit %5.0f px, above limit %2d:%s\n", stem.c_str(), (int)F.boxes.size(),
               (int)F.blobs.size(), limit, over, list.c_str());

        if (!outDir.empty()) {
            // whole frame: blobs above the limit red, 3+ person blobs yellow, other blobs blue
            vector<int> colour(F.blobs.size());
            for (size_t j = 0; j < F.blobs.size(); j++)
                colour[j] = F.info[j].area > limit ? 2 : F.owners[j] >= 3 ? 1 : 0;
            vector<unsigned char> out(F.w * F.h * 3);
            drawRegion(F, 0, 0, F.w - 1, F.h - 1, 1, colour, out, F.w, 0, 0);
            stbi_write_png((outDir + "/clusters_" + stem + ".png").c_str(), F.w, F.h, 3, out.data(), F.w * 3);
        }
    }

    // ---- pictures of up to 12 blobs with 3+ persons ----
    if (!outDir.empty() && !big3.empty()) {
        const int CELL = 120, COLS = 4;
        int n = min((int)big3.size(), 12), rows = (n + COLS - 1) / COLS;
        int W = COLS * CELL, H = rows * CELL;
        vector<unsigned char> out(W * H * 3, 255);
        int step = max(1, (int)big3.size() / n);
        for (int k = 0; k < n; k++) {
            const BlobInfo &bi = big3[k * step];
            Frame F;
            if (!loadFrame(bi.frame, F)) continue;
            int cx = (bi.b.xmin + bi.b.xmax) / 2, cy = (bi.b.ymin + bi.b.ymax) / 2;
            int side = max(bi.longSide + 10, 30), zoom = max(1, (CELL - 4) / side);
            int half = (CELL - 4) / zoom / 2;
            vector<int> colour(F.blobs.size(), -1);
            for (size_t j = 0; j < F.blobs.size(); j++)
                if (F.blobs[j].xmin == bi.b.xmin && F.blobs[j].ymin == bi.b.ymin && F.blobs[j].area == bi.area) colour[j] = 1;
            drawRegion(F, cx - half, cy - half, cx + half - 1, cy + half - 1, zoom, colour, out, W, (k % COLS) * CELL + 2, (k / COLS) * CELL + 2);
        }
        stbi_write_png((outDir + "/clusters_3plus.png").c_str(), W, H, 3, out.data(), W * 3);
        printf("\npictures written to %s\n", outDir.c_str());
    }
    return 0;
}
