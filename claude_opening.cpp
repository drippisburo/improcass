// Analysis by Claude: should morphological OPENING (erosion, then dilation) be used,
// before or after the blob step, or not at all?
//   none   : threshold -> blobs(r) -> drop blobs < MINAREA                       (current pipeline)
//   before : threshold -> opening -> blobs(r) -> drop blobs < MINAREA
//   after  : threshold -> blobs(r) -> drop blobs < MINAREA -> opening -> blobs(r) -> drop < MINAREA
// Structuring elements: 3x3 cross, 3x3 square, 5x5 square.
// Detection: covered (person part of a blob), found (blob centred in the person box), split.
// Errors:    junk blobs (hold no person), pairs far apart merged, pixels removed.
// A person belongs to the blob covering most of their box (label map).
// Same threshold and minimum area as functions.cpp (1.52 x frame mean, blobs >= 4 px).
// Uses claude_functions.cpp; preproc.cpp / functions.cpp are untouched.
//
// build (from albwork\):  g++ -std=c++17 -O2 claude_opening.cpp claude_functions.cpp -o claude_opening.exe
// run   (from albwork\):  .\claude_opening.exe            (text only)
//                         .\claude_opening.exe outdir     (also writes example pictures into outdir)

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
const vector<int> RADII = {1, 2, 6};
const vector<int> ANGLES = {70, 80, 90};

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
int gapGroup(int g) { return g <= 1 ? 0 : g <= 6 ? 1 : g <= 15 ? 2 : 3; }
int band(int alt) { return alt <= 60 ? 0 : alt <= 90 ? 1 : 2; }

// ---- morphology on a 0/1 mask; outside the image counts as background ----
struct SE { const char *name; int k; bool cross; };   // k = reach; cross = only the 4 straight neighbours
const vector<SE> SES = {{"3x3 cross", 1, true}, {"3x3 square", 1, false}, {"5x5 square", 2, false}};

bool inSE(const SE &s, int dx, int dy) { return s.cross ? (dx == 0 || dy == 0) : true; }

vector<char> erode(const vector<char> &m, int w, int h, const SE &s) {
    vector<char> out(w * h, 0);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            if (!m[y * w + x]) continue;
            bool keep = true;
            for (int dy = -s.k; dy <= s.k && keep; dy++)
                for (int dx = -s.k; dx <= s.k; dx++) {
                    if (!inSE(s, dx, dy)) continue;
                    int nx = x + dx, ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h || !m[ny * w + nx]) { keep = false; break; }
                }
            out[y * w + x] = keep;
        }
    return out;
}

vector<char> dilate(const vector<char> &m, int w, int h, const SE &s) {
    vector<char> out(w * h, 0);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            if (!m[y * w + x]) continue;
            for (int dy = -s.k; dy <= s.k; dy++)
                for (int dx = -s.k; dx <= s.k; dx++) {
                    if (!inSE(s, dx, dy)) continue;
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && ny >= 0 && nx < w && ny < h) out[ny * w + nx] = 1;
                }
        }
    return out;
}

vector<char> opening(const vector<char> &m, int w, int h, const SE &s) { return dilate(erode(m, w, h, s), w, h, s); }

// ---- statistics of one pipeline variant ----
struct Stats {
    int covered = 0, found = 0, split = 0, junk = 0, multi = 0, multi3 = 0;
    long long pixels = 0;
    array<int, 3> coveredBand{};
    array<int, 4> merged{};
};

struct Result { vector<Blob> blobs; vector<int> label; };

Result blobsOf(const vector<char> &m, int w, int h, int r) {
    Result R;
    vector<char> work = m;
    R.blobs = claude_findblob((bool *)work.data(), w, h, r, MINAREA, &R.label);
    return R;
}

// all pixels that belong to a kept blob
vector<char> keptMask(const Result &R) {
    vector<char> m(R.label.size(), 0);
    for (size_t i = 0; i < R.label.size(); i++) m[i] = R.label[i] >= 0;
    return m;
}

vector<int> owners(const Result &R, const vector<Box> &boxes, int w) {
    vector<int> own(boxes.size(), -1);
    for (size_t p = 0; p < boxes.size(); p++) {
        const Box &b = boxes[p];
        map<int, int> cnt;
        for (int y = b.ymin; y <= b.ymax; y++)
            for (int x = b.xmin; x <= b.xmax; x++)
                if (R.label[y * w + x] >= 0) cnt[R.label[y * w + x]]++;
        int best = 0;
        for (auto &c : cnt) if (c.second > best) { best = c.second; own[p] = c.first; }
    }
    return own;
}

void addStats(Stats &s, const Result &R, const vector<Box> &boxes, const vector<array<int, 3>> &pairs, int w, int bd) {
    vector<int> own = owners(R, boxes, w);
    vector<int> count(R.blobs.size(), 0);
    for (size_t p = 0; p < boxes.size(); p++) {
        const Box &b = boxes[p];
        int centred = 0;
        for (const Blob &bl : R.blobs)
            if (bl.mx >= b.xmin && bl.mx <= b.xmax && bl.my >= b.ymin && bl.my <= b.ymax) centred++;
        if (centred >= 1) s.found++;
        if (centred >= 2) s.split++;
        if (own[p] >= 0) { s.covered++; s.coveredBand[bd]++; count[own[p]]++; }
    }
    for (size_t j = 0; j < R.blobs.size(); j++) {
        if (count[j] == 0) s.junk++;
        if (count[j] >= 2) s.multi++;
        if (count[j] >= 3) s.multi3++;
        s.pixels += R.blobs[j].area;
    }
    for (auto &q : pairs)
        if (own[q[0]] >= 0 && own[q[0]] == own[q[1]]) s.merged[q[2]]++;
}

// picture: 3 panels (image | mask without opening | mask after opening), person boxes green
void writeExample(const string &file, const vector<unsigned char> &img, const vector<char> &a, const vector<char> &b,
                  int w, const vector<Box> &boxes, int x0, int y0, int x1, int y1, int zoom) {
    int cw = x1 - x0 + 1, ch = y1 - y0 + 1, G = 6, W = 3 * cw * zoom + 2 * G;
    vector<unsigned char> out(W * ch * zoom * 3, 255);
    for (int panel = 0; panel < 3; panel++)
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++) {
                int i = y * w + x;
                unsigned char r = img[i], g = img[i], bl = img[i];
                const vector<char> *m = panel == 1 ? &a : panel == 2 ? &b : nullptr;
                if (m && (*m)[i]) { r = 255; g = 40; bl = 40; }
                for (const Box &bx : boxes) {
                    bool e = ((x == bx.xmin || x == bx.xmax) && y >= bx.ymin && y <= bx.ymax) ||
                             ((y == bx.ymin || y == bx.ymax) && x >= bx.xmin && x <= bx.xmax);
                    if (e) { r = 0; g = 230; bl = 0; }
                }
                for (int zy = 0; zy < zoom; zy++)
                    for (int zx = 0; zx < zoom; zx++) {
                        int o = (((y - y0) * zoom + zy) * W + panel * (cw * zoom + G) + (x - x0) * zoom + zx) * 3;
                        out[o] = r; out[o + 1] = g; out[o + 2] = bl;
                    }
            }
    stbi_write_png(file.c_str(), W, ch * zoom, 3, out.data(), W * 3);
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

    // variant index: [place][se][r]; place 0 = none (se unused), 1 = before, 2 = after
    int nS = SES.size(), nR = RADII.size();
    vector<Stats> st(3 * nS * nR);
    auto idx = [&](int place, int se, int r) { return (place * nS + se) * nR + r; };
    array<int, 4> pairsPerGroup{};
    array<int, 3> personsBand{};
    int persons = 0;
    bool wroteSmall = false, wroteJunk = false;

    for (const string &stem : frames) {
        vector<Box> boxes = personBoxes(XML_DIR + stem + ".xml");
        int w, h, channels;
        unsigned char *data = stbi_load((IMG_DIR + stem + ".jpg").c_str(), &w, &h, &channels, 1);
        if (!data) { cout << "no image was loaded: " << stem << endl; continue; }
        vector<unsigned char> img(data, data + w * h);
        stbi_image_free(data);
        for (Box &b : boxes) {
            b.xmin = max(0, b.xmin); b.ymin = max(0, b.ymin);
            b.xmax = min(w - 1, b.xmax); b.ymax = min(h - 1, b.ymax);
        }
        int bd = band(field(stem, 1));
        persons += boxes.size();
        personsBand[bd] += boxes.size();
        vector<array<int, 3>> pairs;
        for (size_t i = 0; i < boxes.size(); i++)
            for (size_t j = i + 1; j < boxes.size(); j++) {
                int g = gapGroup(gapBetween(boxes[i], boxes[j]));
                pairs.push_back({(int)i, (int)j, g});
                pairsPerGroup[g]++;
            }

        double sum = 0;
        for (int i = 0; i < w * h; i++) sum += img[i];
        float mean = sum / (w * h);
        vector<char> mask(w * h);
        for (int i = 0; i < w * h; i++) mask[i] = claude_hotpixmask((float)img[i], mean, BRIGHT);

        vector<vector<char>> openedBefore(nS);
        for (int s = 0; s < nS; s++) openedBefore[s] = opening(mask, w, h, SES[s]);

        for (int r = 0; r < nR; r++) {
            Result none = blobsOf(mask, w, h, RADII[r]);
            for (int s = 0; s < nS; s++) {
                // the "none" variant is the same for every se; store it once (se 0)
                if (s == 0) addStats(st[idx(0, 0, r)], none, boxes, pairs, w, bd);
                Result before = blobsOf(openedBefore[s], w, h, RADII[r]);
                addStats(st[idx(1, s, r)], before, boxes, pairs, w, bd);
                Result after = blobsOf(opening(keptMask(none), w, h, SES[s]), w, h, RADII[r]);
                addStats(st[idx(2, s, r)], after, boxes, pairs, w, bd);

                // example pictures (r = 2, 3x3 square before): a person lost by opening, and junk removed
                if (!outDir.empty() && RADII[r] == 2 && s == 1) {
                    vector<int> ownNone = owners(none, boxes, w), ownBefore = owners(before, boxes, w);
                    for (size_t p = 0; p < boxes.size() && !wroteSmall; p++)
                        if (ownNone[p] >= 0 && ownBefore[p] < 0 && field(stem, 1) >= 100 && boxes.size() >= 4) {
                            const Box &b = boxes[p];
                            int cx = (b.xmin + b.xmax) / 2, cy = (b.ymin + b.ymax) / 2;
                            writeExample(outDir + "/opening_lost_person.png", img, keptMask(none), keptMask(before), w, boxes,
                                         max(0, cx - 30), max(0, cy - 30), min(w - 1, cx + 30), min(h - 1, cy + 30), 5);
                            printf("example picture: person lost by opening in %s, box (%d,%d)-(%d,%d)\n", stem.c_str(), b.xmin, b.ymin, b.xmax, b.ymax);
                            wroteSmall = true;
                        }
                    int junkNone = 0, junkBefore = 0;
                    vector<int> cN(none.blobs.size(), 0), cB(before.blobs.size(), 0);
                    for (int o : ownNone) if (o >= 0) cN[o]++;
                    for (int o : ownBefore) if (o >= 0) cB[o]++;
                    for (int c : cN) if (c == 0) junkNone++;
                    for (int c : cB) if (c == 0) junkBefore++;
                    if (!wroteJunk && junkNone - junkBefore >= 25) {
                        writeExample(outDir + "/opening_junk.png", img, keptMask(none), keptMask(before), w, boxes, 0, 0, w - 1, h - 1, 1);
                        printf("example picture: junk removed by opening in %s (%d -> %d junk blobs)\n", stem.c_str(), junkNone, junkBefore);
                        wroteJunk = true;
                    }
                }
            }
        }
    }

    int nf = frames.size();
    printf("\nthreshold %.2f x frame mean, blobs >= %d px, %d frames, %d persons\n", BRIGHT, MINAREA, nf, persons);
    printf("covered = person part of a blob, found = blob centred in the box, junk = blobs holding no person,\n");
    printf("touching / 7-15px = share of such pairs in one blob, >15px = number of such pairs in one blob\n\n");
    printf("%-26s %3s %8s %8s %7s %9s %8s %8s %9s %8s %6s %6s %8s\n", "variant", "r", "covered", "found", "split", "junk/fr",
           "multi", "3+", "touching", "7-15px", ">15px", "", "pixels");
    for (int r = 0; r < nR; r++) {
        for (int place = 0; place < 3; place++)
            for (int s = 0; s < nS; s++) {
                if (place == 0 && s > 0) continue;
                Stats &x = st[idx(place, s, r)];
                string name = place == 0 ? "none" : string(place == 1 ? "before, " : "after,  ") + SES[s].name;
                long long base = st[idx(0, 0, r)].pixels;
                printf("%-26s %3d %7.2f%% %7.2f%% %6.1f%% %9.1f %8d %8d %8.1f%% %7.1f%% %6d %6s %7.0f%%\n", name.c_str(), RADII[r],
                       100.0 * x.covered / persons, 100.0 * x.found / persons, 100.0 * x.split / persons, (double)x.junk / nf,
                       x.multi, x.multi3, 100.0 * x.merged[0] / pairsPerGroup[0], 100.0 * x.merged[2] / pairsPerGroup[2],
                       x.merged[3], "", 100.0 * x.pixels / base);
            }
        printf("\n");
    }

    printf("covered by altitude band (r = 2)\n%-26s %10s %10s %10s\n", "variant", "60m", "70-90m", "100-130m");
    int r2 = 1;
    for (int place = 0; place < 3; place++)
        for (int s = 0; s < nS; s++) {
            if (place == 0 && s > 0) continue;
            Stats &x = st[idx(place, s, r2)];
            string name = place == 0 ? "none" : string(place == 1 ? "before, " : "after,  ") + SES[s].name;
            printf("%-26s", name.c_str());
            for (int b = 0; b < 3; b++) printf(" %9.1f%%", 100.0 * x.coveredBand[b] / personsBand[b]);
            printf("\n");
        }
    return 0;
}
