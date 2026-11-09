#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <random>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;
namespace fs = std::filesystem;

const string IMG_DIR = "../split_daynight/night/JPEGImages/";
const string XML_DIR = "../split_daynight/night/Annotations/";

// Step 1: compare a person's pixel values to their surroundings and to the whole frame.
// Run from albwork\ so the relative image path resolves.

struct Box { int xmin, ymin, xmax, ymax; };

void explore(const char *name, Box b) {
    string path = string("../split_daynight/night/JPEGImages/") + name;
    int w, h, channels;
    unsigned char *data = stbi_load(path.c_str(), &w, &h, &channels, 1);   // force 1 channel
    if (!data) { cout << "no image was loaded: " << path << endl; return; }

    float *f = new float[w * h];
    for (int i = 0; i < w * h; i++) f[i] = (float)data[i];

    // whole-frame mean and population std
    double sum = 0, sumSq = 0;
    for (int i = 0; i < w * h; i++) { sum += f[i]; sumSq += (double)f[i] * f[i]; }
    double gMean = sum / (w * h);
    double gStd = sqrt(sumSq / (w * h) - gMean * gMean);

    // inside the person box
    float inMin = 255, inMax = 0;
    double inSum = 0;
    int inN = 0;
    for (int y = b.ymin; y <= b.ymax; y++) {
        for (int x = b.xmin; x <= b.xmax; x++) {
            float v = f[y * w + x];
            if (v < inMin) inMin = v;
            if (v > inMax) inMax = v;
            inSum += v;
            inN++;
        }
    }

    // surroundings: ring of pixels within `margin` of the box, excluding the box itself
    int margin = 15;
    int x0 = max(0, b.xmin - margin), x1 = min(w - 1, b.xmax + margin);
    int y0 = max(0, b.ymin - margin), y1 = min(h - 1, b.ymax + margin);
    double ringSum = 0;
    int ringN = 0;
    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            bool inside = x >= b.xmin && x <= b.xmax && y >= b.ymin && y <= b.ymax;
            if (inside) continue;
            ringSum += f[y * w + x];
            ringN++;
        }
    }
    double ringMean = ringSum / ringN;

    cout << name << "  (box " << b.xmax - b.xmin + 1 << "x" << b.ymax - b.ymin + 1 << " px)" << endl;
    cout << "  person   min " << inMin << "  max " << inMax << "  mean " << inSum / inN << endl;
    cout << "  ring     mean " << ringMean << endl;
    cout << "  frame    mean " << gMean << "  std " << gStd << endl;
    cout << "  peak - ring mean = " << inMax - ringMean << "   peak vs frame mean + 3*std = "
         << inMax << " vs " << gMean + 3 * gStd << endl << endl;

    delete[] f;
    stbi_image_free(data);
}

// ---------------------------------------------------------------------------
// Step 1b: is a simple "brighter than x% of the frame" threshold enough?
// For each frame: background = all pixels outside every person box.
// For each person box: mean of its brightest 25% pixels, ratio to background mean,
// and percentile (% of frame pixels darker than that top-25% mean).
// ---------------------------------------------------------------------------

// integer between <tag> and </tag>, searching from position `from`
int tagInt(const string &s, const string &tag, size_t from) {
    size_t a = s.find("<" + tag + ">", from) + tag.size() + 2;
    return stoi(s.substr(a, s.find("</" + tag + ">", a) - a));
}

// all Person boxes in one Pascal VOC XML file
vector<Box> personBoxes(const string &xmlName) {
    ifstream in(XML_DIR + xmlName);
    stringstream ss;
    ss << in.rdbuf();
    string s = ss.str();

    vector<Box> boxes;
    size_t pos = 0;
    while ((pos = s.find("<object>", pos)) != string::npos) {
        size_t n = s.find("<name>", pos) + 6;
        string name = s.substr(n, s.find("</name>", n) - n);
        if (name == "Person")
            boxes.push_back({tagInt(s, "xmin", pos), tagInt(s, "ymin", pos),
                             tagInt(s, "xmax", pos), tagInt(s, "ymax", pos)});
        pos = s.find("</object>", pos);
    }
    return boxes;
}

// field i (0-based) of daynight_altitude_angle_0_frame, e.g. field 2 = angle
int field(const string &name, int i) {
    stringstream ss(name);
    string t;
    for (int k = 0; k <= i; k++) getline(ss, t, '_');
    return stoi(t);
}

// `count` random frames (names without extension) with one of `angles` and at least one person
vector<string> pickRandom(const vector<int> &angles, int count, mt19937 &rng) {
    vector<string> pool;
    for (auto &e : fs::directory_iterator(XML_DIR)) {
        string xml = e.path().filename().string();
        int a = field(xml, 2);
        if (find(angles.begin(), angles.end(), a) == angles.end()) continue;
        if (personBoxes(xml).empty()) continue;
        pool.push_back(xml.substr(0, xml.size() - 4));   // drop ".xml"
    }
    sort(pool.begin(), pool.end());   // directory order isn't guaranteed; sort so the seed is reproducible
    shuffle(pool.begin(), pool.end(), rng);
    if ((int)pool.size() > count) pool.resize(count);
    return pool;
}

const bool SHOW_BOXES = false;   // true: print one line per person box

// one person box: its top-25% mean divided by the background mean / by the whole-frame mean
struct PersonRatio { double bg, frame; string where; };

struct Stats {
    vector<PersonRatio> persons;
    int n;                        // number of person boxes
    double frameMean, bgMean;     // whole frame, frame without person boxes
    double personMean, minPerson; // mean of the box means, darkest box mean
    double ratio, minRatio;       // box mean / background mean
    double pct, minPct;           // % of frame pixels darker than the box mean
};

Stats comparePersons(const string &stem) {
    Stats st = {};
    vector<Box> boxes = personBoxes(stem + ".xml");
    int w, h, channels;
    unsigned char *data = stbi_load((IMG_DIR + stem + ".jpg").c_str(), &w, &h, &channels, 1);
    if (!data) { cout << "no image was loaded: " << stem << endl; return st; }

    float *f = new float[w * h];
    for (int i = 0; i < w * h; i++) f[i] = (float)data[i];

    // mark every pixel covered by a person box (boxes clamped to the image)
    vector<bool> inBox(w * h, false);
    for (Box &b : boxes) {
        b.xmin = max(0, b.xmin); b.ymin = max(0, b.ymin);
        b.xmax = min(w - 1, b.xmax); b.ymax = min(h - 1, b.ymax);
        for (int y = b.ymin; y <= b.ymax; y++)
            for (int x = b.xmin; x <= b.xmax; x++)
                inBox[y * w + x] = true;
    }

    double frameSum = 0, bgSum = 0;
    int bgN = 0;
    for (int i = 0; i < w * h; i++) {
        frameSum += f[i];
        if (!inBox[i]) { bgSum += f[i]; bgN++; }
    }
    st.frameMean = frameSum / (w * h);
    st.bgMean = bgSum / bgN;

    // sorted copy of the frame, to look up percentiles
    vector<float> sorted(f, f + w * h);
    sort(sorted.begin(), sorted.end());

    st.n = boxes.size();
    st.minPerson = 255; st.minRatio = 1e9; st.minPct = 100;
    string boxLines;
    for (Box &b : boxes) {
        // brightest 25% of the box pixels (at least 1), brightest first
        vector<float> vals;
        for (int y = b.ymin; y <= b.ymax; y++)
            for (int x = b.xmin; x <= b.xmax; x++) vals.push_back(f[y * w + x]);
        sort(vals.begin(), vals.end(), greater<float>());
        int top = max(1, (int)vals.size() / 4);
        double sum = 0;
        for (int i = 0; i < top; i++) sum += vals[i];
        double m = sum / top;

        char line[96];
        snprintf(line, sizeof line, "    box %3dx%-3d top %3d px: %3.0f..%3.0f  mean %5.1f\n",
                 b.xmax - b.xmin + 1, b.ymax - b.ymin + 1, top, vals[top - 1], vals[0], m);
        boxLines += line;
        double ratio = m / st.bgMean;
        double pct = 100.0 * (lower_bound(sorted.begin(), sorted.end(), (float)m) - sorted.begin()) / (w * h);
        st.persons.push_back({ratio, m / st.frameMean,
                              stem + " box " + to_string(b.xmax - b.xmin + 1) + "x" + to_string(b.ymax - b.ymin + 1)});

        st.personMean += m;  st.ratio += ratio;  st.pct += pct;
        st.minPerson = min(st.minPerson, m);
        st.minRatio = min(st.minRatio, ratio);
        st.minPct = min(st.minPct, pct);
    }
    st.personMean /= st.n;  st.ratio /= st.n;  st.pct /= st.n;

    printf("%-24s %3d %7.1f %7.1f %7.1f %7.1f   %5.2f %5.2f   %5.1f %5.1f\n",
           stem.c_str(), st.n, st.frameMean, st.bgMean, st.personMean, st.minPerson,
           st.ratio, st.minRatio, st.pct, st.minPct);
    if (SHOW_BOXES) cout << boxLines;

    delete[] f;
    stbi_image_free(data);
    return st;
}

// how low the multiplier would have to be to keep every person in `all`
void summariseRatios(const char *label, vector<PersonRatio> all) {
    sort(all.begin(), all.end(), [](const PersonRatio &a, const PersonRatio &b) { return a.bg < b.bg; });
    double minFrame = 1e9;
    for (const PersonRatio &p : all) minFrame = min(minFrame, p.frame);

    printf("\n-- %s: %d persons\n", label, (int)all.size());
    printf("   lowest multiplier: %.2f x background mean  (%.2f x whole-frame mean)\n", all[0].bg, minFrame);
    printf("   5 lowest:\n");
    for (int i = 0; i < 5 && i < (int)all.size(); i++)
        printf("     %.2f  %s\n", all[i].bg, all[i].where.c_str());
    printf("   persons kept at multiplier:");
    for (double t : {1.3, 1.4, 1.5, 1.6, 1.7}) {
        int kept = 0;
        for (const PersonRatio &p : all) if (p.bg > t) kept++;
        printf("  %.1f -> %.1f%%", t, 100.0 * kept / all.size());
    }
    printf("\n");
}

vector<PersonRatio> runGroup(const char *label, const vector<int> &angles, int count, mt19937 &rng) {
    cout << endl << "== " << label << endl;
    printf("%-24s %3s %7s %7s %7s %7s   %5s %5s   %5s %5s\n", "frame", "n", "frame", "bg",
           "person", "minP", "ratio", "minR", "pct", "minPc");

    vector<string> frames = pickRandom(angles, count, rng);
    vector<PersonRatio> all;
    for (const string &s : frames) {
        Stats st = comparePersons(s);
        all.insert(all.end(), st.persons.begin(), st.persons.end());
    }
    summariseRatios(label, all);
    return all;
}

// ---------------------------------------------------------------------------
// Step 1c: threshold at MULT x whole-frame mean and look at what survives.
// Saves an overlay per frame to OUT_DIR: kept pixels red, person boxes green.
// ---------------------------------------------------------------------------

const double MULT = 1.3;
const int MIN_AREA = 6;          // blobs smaller than this count as noise
const string OUT_DIR = "masks/";

struct Blob { int area, xmin, ymin, xmax, ymax; double cx, cy; };

// 8-connected components of `mask`, flood fill with an explicit stack.
// If `labels` is given, it is filled with the blob index of every pixel (-1 = background).
vector<Blob> findBlobs(const vector<bool> &mask, int w, int h, vector<int> *labels = nullptr) {
    if (labels) labels->assign(w * h, -1);
    vector<bool> seen(w * h, false);
    vector<Blob> blobs;
    vector<int> stack;
    for (int start = 0; start < w * h; start++) {
        if (!mask[start] || seen[start]) continue;
        Blob b = {0, w, h, -1, -1, 0, 0};
        seen[start] = true;
        stack.push_back(start);
        while (!stack.empty()) {
            int i = stack.back();
            stack.pop_back();
            if (labels) (*labels)[i] = blobs.size();
            int x = i % w, y = i / w;
            b.area++;  b.cx += x;  b.cy += y;
            b.xmin = min(b.xmin, x);  b.xmax = max(b.xmax, x);
            b.ymin = min(b.ymin, y);  b.ymax = max(b.ymax, y);
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx, ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    int j = ny * w + nx;
                    if (mask[j] && !seen[j]) { seen[j] = true; stack.push_back(j); }
                }
        }
        b.cx /= b.area;  b.cy /= b.area;
        blobs.push_back(b);
    }
    return blobs;
}

struct MaskStats { int persons, touched, hit, blobs, bigBlobs, extra; };

MaskStats maskTest(const string &stem) {
    MaskStats ms = {};
    vector<Box> boxes = personBoxes(stem + ".xml");
    int w, h, channels;
    unsigned char *data = stbi_load((IMG_DIR + stem + ".jpg").c_str(), &w, &h, &channels, 1);
    if (!data) { cout << "no image was loaded: " << stem << endl; return ms; }

    double sum = 0;
    for (int i = 0; i < w * h; i++) sum += data[i];
    double mean = sum / (w * h);
    double thresh = MULT * mean;

    vector<bool> mask(w * h);
    int kept = 0;
    for (int i = 0; i < w * h; i++) {
        mask[i] = data[i] > thresh;
        if (mask[i]) kept++;
    }

    vector<Blob> blobs = findBlobs(mask, w, h);
    int largest = 0;
    for (const Blob &b : blobs) {
        largest = max(largest, b.area);
        if (b.area >= MIN_AREA) ms.bigBlobs++;
    }
    ms.blobs = blobs.size();

    // person found: "touched" = any kept pixel in its box, "hit" = a blob centroid in its box
    auto inside = [](const Box &b, double x, double y) {
        return x >= b.xmin && x <= b.xmax && y >= b.ymin && y <= b.ymax;
    };
    for (Box &b : boxes) {
        b.xmin = max(0, b.xmin); b.ymin = max(0, b.ymin);
        b.xmax = min(w - 1, b.xmax); b.ymax = min(h - 1, b.ymax);
        bool touched = false, hit = false;
        for (int y = b.ymin; y <= b.ymax && !touched; y++)
            for (int x = b.xmin; x <= b.xmax; x++)
                if (mask[y * w + x]) { touched = true; break; }
        for (const Blob &bl : blobs)
            if (inside(b, bl.cx, bl.cy)) { hit = true; break; }
        ms.touched += touched;
        ms.hit += hit;
    }
    ms.persons = boxes.size();

    // blobs (>= MIN_AREA) whose centroid is in no person box -> would become clutter
    for (const Blob &bl : blobs) {
        if (bl.area < MIN_AREA) continue;
        bool any = false;
        for (const Box &b : boxes) if (inside(b, bl.cx, bl.cy)) { any = true; break; }
        if (!any) ms.extra++;
    }

    // overlay: gray frame, kept pixels red, person boxes green
    vector<unsigned char> rgb(w * h * 3);
    for (int i = 0; i < w * h; i++) {
        unsigned char g = data[i];
        if (mask[i]) { rgb[3*i] = 255; rgb[3*i+1] = g / 4; rgb[3*i+2] = g / 4; }
        else         { rgb[3*i] = g;   rgb[3*i+1] = g;     rgb[3*i+2] = g; }
    }
    for (const Box &b : boxes)
        for (int y = b.ymin; y <= b.ymax; y++)
            for (int x = b.xmin; x <= b.xmax; x++) {
                if (x != b.xmin && x != b.xmax && y != b.ymin && y != b.ymax) continue;
                int i = y * w + x;
                rgb[3*i] = 0; rgb[3*i+1] = 255; rgb[3*i+2] = 0;
            }
    string out = OUT_DIR + stem + "_mask.png";
    if (!stbi_write_png(out.c_str(), w, h, 3, rgb.data(), w * 3))
        cout << "could not write " << out << endl;

    printf("%-20s %6.1f %6.1f %5.1f%% %6d %6d %6d   %3d %4d %4d %6d\n",
           stem.c_str(), mean, thresh, 100.0 * kept / (w * h), ms.blobs, ms.bigBlobs, largest,
           ms.persons, ms.touched, ms.hit, ms.extra);

    stbi_image_free(data);
    return ms;
}

void runMaskTest(const vector<string> &frames) {
    printf("threshold = %.2f x frame mean, blobs >= %d px count as real\n\n", MULT, MIN_AREA);
    printf("%-20s %6s %6s %6s %6s %6s %6s   %3s %4s %4s %6s\n", "frame", "mean", "thresh",
           "kept", "blobs", ">=min", "maxA", "ppl", "tch", "hit", "extra");
    MaskStats total = {};
    for (const string &s : frames) {
        MaskStats ms = maskTest(s);
        total.persons += ms.persons;  total.touched += ms.touched;  total.hit += ms.hit;
        total.bigBlobs += ms.bigBlobs;  total.extra += ms.extra;
    }
    printf("\ntotal: %d persons, touched %d (%.1f%%), centroid hit %d (%.1f%%), "
           "%.1f blobs >= %d px per frame, %.1f of them extra (no person)\n",
           total.persons, total.touched, 100.0 * total.touched / total.persons,
           total.hit, 100.0 * total.hit / total.persons,
           (double)total.bigBlobs / frames.size(), MIN_AREA, (double)total.extra / frames.size());
}

// ---------------------------------------------------------------------------
// Step 1d: for every person box, brightest pixel / whole-frame mean.
// A person survives "pixel >= factor * frame mean" if at least one of their
// pixels passes, i.e. if box max / frame mean >= factor. So the smallest ratio
// over all persons is the highest factor that still keeps every person.
// ---------------------------------------------------------------------------

const double FACTOR = 1.5;   // the factor being judged; change freely
const vector<double> FACTOR_TABLE = {1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0};

struct MaxRatio { double r; string where; int angle; };

void maxRatioStudy(const vector<int> &angles) {
    mt19937 rng(0);
    vector<string> frames = pickRandom(angles, 1 << 30, rng);   // all matching frames

    vector<MaxRatio> all;
    vector<double> keptPct(FACTOR_TABLE.size(), 0);   // % of frame pixels passing, summed over frames
    for (const string &stem : frames) {
        vector<Box> boxes = personBoxes(stem + ".xml");
        int w, h, channels;
        unsigned char *data = stbi_load((IMG_DIR + stem + ".jpg").c_str(), &w, &h, &channels, 1);
        if (!data) { cout << "no image was loaded: " << stem << endl; continue; }

        double sum = 0;
        for (int i = 0; i < w * h; i++) sum += data[i];
        double mean = sum / (w * h);

        for (size_t k = 0; k < FACTOR_TABLE.size(); k++) {
            int kept = 0;
            for (int i = 0; i < w * h; i++) if (data[i] >= FACTOR_TABLE[k] * mean) kept++;
            keptPct[k] += 100.0 * kept / (w * h);
        }

        for (Box b : boxes) {
            b.xmin = max(0, b.xmin); b.ymin = max(0, b.ymin);
            b.xmax = min(w - 1, b.xmax); b.ymax = min(h - 1, b.ymax);
            int peak = 0;
            for (int y = b.ymin; y <= b.ymax; y++)
                for (int x = b.xmin; x <= b.xmax; x++) peak = max(peak, (int)data[y * w + x]);
            all.push_back({peak / mean,
                           stem + " box " + to_string(b.xmax - b.xmin + 1) + "x" + to_string(b.ymax - b.ymin + 1)
                               + "  peak " + to_string(peak) + "  frame mean " + to_string((int)round(mean)),
                           field(stem, 2)});
        }
        stbi_image_free(data);
    }
    sort(all.begin(), all.end(), [](const MaxRatio &a, const MaxRatio &b) { return a.r < b.r; });

    printf("%d frames, %d persons (angles", (int)frames.size(), (int)all.size());
    for (int a : angles) printf(" %d", a);
    printf(")\n\n");
    printf("highest factor that keeps EVERY person: %.3f\n", all[0].r);
    for (int a : angles) {
        for (const MaxRatio &m : all)
            if (m.angle == a) { printf("   angle %d: %.3f\n", a, m.r); break; }
    }

    printf("\n10 weakest persons (box max / frame mean):\n");
    for (int i = 0; i < 10 && i < (int)all.size(); i++)
        printf("   %.3f  %s\n", all[i].r, all[i].where.c_str());

    printf("\n%-8s %10s %8s %16s\n", "factor", "persons", "missed", "frame px kept");
    for (size_t k = 0; k < FACTOR_TABLE.size(); k++) {
        double f = FACTOR_TABLE[k];
        int missed = 0;
        for (const MaxRatio &m : all) if (m.r < f) missed++;
        printf("%-8.2f %9.2f%% %8d %15.2f%%%s\n", f, 100.0 * (all.size() - missed) / all.size(), missed,
               keptPct[k] / frames.size(), fabs(f - FACTOR) < 1e-9 ? "   <- FACTOR" : "");
    }
}

// ---------------------------------------------------------------------------
// Step 1e: threshold at BLOB_FACTOR x frame mean, find 8-connected blobs, and
// count how many persons survive when blobs smaller than N px are dropped.
// kept    = a blob of >= N px has its centroid inside the person box
// touched = a blob of >= N px has at least one pixel inside the person box
// ---------------------------------------------------------------------------

const double BLOB_FACTOR = 1.54;
const vector<int> MIN_AREAS = {1, 3, 5, 10, 15, 20, 25, 30};

struct PersonBlob { int centroidArea, touchArea, altitude; };

void blobSizeStudy(const vector<int> &angles) {
    mt19937 rng(0);
    vector<string> frames = pickRandom(angles, 1 << 30, rng);   // all matching frames

    vector<PersonBlob> persons;
    vector<int> blobsAtLeast(MIN_AREAS.size(), 0), extraAtLeast(MIN_AREAS.size(), 0);
    for (const string &stem : frames) {
        vector<Box> boxes = personBoxes(stem + ".xml");
        int w, h, channels;
        unsigned char *data = stbi_load((IMG_DIR + stem + ".jpg").c_str(), &w, &h, &channels, 1);
        if (!data) { cout << "no image was loaded: " << stem << endl; continue; }

        double sum = 0;
        for (int i = 0; i < w * h; i++) sum += data[i];
        double thresh = BLOB_FACTOR * sum / (w * h);
        vector<bool> mask(w * h);
        for (int i = 0; i < w * h; i++) mask[i] = data[i] >= thresh;

        vector<int> label;
        vector<Blob> blobs = findBlobs(mask, w, h, &label);

        for (Box &b : boxes) {
            b.xmin = max(0, b.xmin); b.ymin = max(0, b.ymin);
            b.xmax = min(w - 1, b.xmax); b.ymax = min(h - 1, b.ymax);
        }
        auto inside = [](const Box &b, double x, double y) {
            return x >= b.xmin && x <= b.xmax && y >= b.ymin && y <= b.ymax;
        };

        // blobs whose centroid is in no person box would become clutter
        vector<bool> hasPerson(blobs.size(), false);
        for (size_t j = 0; j < blobs.size(); j++)
            for (const Box &b : boxes)
                if (inside(b, blobs[j].cx, blobs[j].cy)) { hasPerson[j] = true; break; }
        for (size_t k = 0; k < MIN_AREAS.size(); k++)
            for (size_t j = 0; j < blobs.size(); j++)
                if (blobs[j].area >= MIN_AREAS[k]) {
                    blobsAtLeast[k]++;
                    if (!hasPerson[j]) extraAtLeast[k]++;
                }

        // per person: largest blob centred in the box, largest blob touching the box
        for (const Box &b : boxes) {
            int cA = 0, tA = 0;
            for (const Blob &bl : blobs)
                if (inside(b, bl.cx, bl.cy)) cA = max(cA, bl.area);
            for (int y = b.ymin; y <= b.ymax; y++)
                for (int x = b.xmin; x <= b.xmax; x++) {
                    int l = label[y * w + x];
                    if (l >= 0) tA = max(tA, blobs[l].area);
                }
            persons.push_back({cA, tA, field(stem, 1)});
        }
        stbi_image_free(data);
    }

    int n = persons.size(), nf = frames.size();
    printf("threshold %.2f x frame mean, %d frames, %d persons (angles", BLOB_FACTOR, nf, n);
    for (int a : angles) printf(" %d", a);
    printf(")\n\n");
    printf("%-8s %16s %16s %12s %14s\n", "min px", "kept (centroid)", "touched", "blobs/frame", "extra/frame");
    for (size_t k = 0; k < MIN_AREAS.size(); k++) {
        int kept = 0, touched = 0;
        for (const PersonBlob &p : persons) {
            if (p.centroidArea >= MIN_AREAS[k]) kept++;
            if (p.touchArea >= MIN_AREAS[k]) touched++;
        }
        printf("%-8d %7.2f%% (%4d) %7.2f%% (%4d) %12.1f %14.1f\n", MIN_AREAS[k],
               100.0 * kept / n, n - kept, 100.0 * touched / n, n - touched,
               (double)blobsAtLeast[k] / nf, (double)extraAtLeast[k] / nf);
    }
    printf("(numbers in brackets = persons missed)\n");

    // kept (centroid) split by altitude: small people at high altitude go first
    vector<int> alts;
    for (const PersonBlob &p : persons)
        if (find(alts.begin(), alts.end(), p.altitude) == alts.end()) alts.push_back(p.altitude);
    sort(alts.begin(), alts.end());
    printf("\nkept (centroid) by altitude\n%-8s", "min px");
    for (int a : alts) printf(" %7dm", a);
    printf("\n%-8s", "persons");
    for (int a : alts) {
        int c = 0;
        for (const PersonBlob &p : persons) if (p.altitude == a) c++;
        printf(" %8d", c);
    }
    printf("\n");
    for (int m : MIN_AREAS) {
        printf("%-8d", m);
        for (int a : alts) {
            int c = 0, kept = 0;
            for (const PersonBlob &p : persons)
                if (p.altitude == a) { c++; if (p.centroidArea >= m) kept++; }
            printf(" %7.1f%%", 100.0 * kept / c);
        }
        printf("\n");
    }
}

int main(void) {
    // Step 1a (single hand-picked boxes):
    // explore("1_60_30_0_00000.jpg",  {271, 72, 278, 88});
    // explore("1_60_30_0_00000.jpg",  {275, 148, 285, 167});
    // explore("1_90_30_0_01370.jpg",  {218, 124, 227, 142});
    // explore("1_130_30_0_03807.jpg", {332, 23, 339, 34});

    unsigned seed = 42;   // change to draw a different random set
    mt19937 rng(seed);
    cout << "seed " << seed << endl;

    // Step 1b (top-25% ratios over many frames):
    // vector<PersonRatio> a = runGroup("angle 80/90 deg", {80, 90}, 40, rng);
    // vector<PersonRatio> b = runGroup("angle 70 deg", {70}, 30, rng);
    // a.insert(a.end(), b.begin(), b.end());
    // summariseRatios("all frames", a);

    // Step 1c: random frames plus the weakest / warmest frames found in step 1b
    // fs::create_directories(OUT_DIR);
    // vector<string> frames = pickRandom({80, 90}, 5, rng);
    // vector<string> f70 = pickRandom({70}, 3, rng);
    // frames.insert(frames.end(), f70.begin(), f70.end());
    // frames.push_back("1_60_80_0_00672");   // lowest ratio, 1.27
    // frames.push_back("1_90_80_0_05034");   // warm background
    // frames.push_back("1_90_70_0_01522");   // warm background
    // runMaskTest(frames);

    // Step 1d: box max / frame mean over every frame at these angles
    // maxRatioStudy({70, 80, 90});

    // Step 1e: how many persons survive a minimum blob size
    blobSizeStudy({70, 80, 90});
    return 0;
}
