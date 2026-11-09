#pragma once
#include <vector>

// Copy of functions.h made by Claude for claude_radius.cpp.
// The blob radius and minimum area are parameters here so the analysis can loop over them.
// The original functions.h / functions.cpp are not touched.

struct Blob {
    int area;
    int xmin, ymin, xmax, ymax;
    float mx, my;
};

bool claude_hotpixmask(float pix, float mean, float bright_threshold);

// same as findblob, but r and minarea are parameters.
// labels (optional): filled with the index of the kept blob each pixel belongs to, -1 for the rest
std::vector<Blob> claude_findblob(bool *mask, int w, int h, int r, int minarea, std::vector<int> *labels = nullptr);
