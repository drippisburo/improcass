#include <stdio.h>
#include <iostream>
#include <math.h>
#include "claude_functions.h"
#include <vector>

// Copy of functions.cpp made by Claude for claude_radius.cpp.
// Only change: r and minarea are parameters, and kept blobs can be written into a label map.

using namespace std;

bool claude_hotpixmask(float pix, float mean, float bright_threshold){
    return (pix >= bright_threshold * mean) ? true : false;
}

vector<Blob> claude_findblob(bool *mask, int w, int h, int r, int minarea, vector<int> *labels) {
    vector<Blob> blobs;      // every blob found in the frame in here
    vector<int> todo;        // pixels of the current blob that still need a look around
    vector<int> members;     // pixels of the current blob (only needed for the label map)
    if (labels) labels->assign(w * h, -1);

    for (int y = 0; y < h; y++){
        for (int x = 0; x < w; x++){
            if (mask[y*w + x]) {
                // a new blob starts here
                Blob hi;
                hi.area = 0;
                hi.xmin = w;  hi.ymin = h;
                hi.xmax = -1; hi.ymax = -1;
                double sumx = 0, sumy = 0;
                members.clear();

                mask[y*w + x] = false;    // off so not counted no more
                todo.push_back(y*w + x);

                while (!todo.empty()) {
                    int now = todo.back();
                    todo.pop_back();
                    int nowx = now % w;
                    int nowy = now / w;
                    if (labels) members.push_back(now);

                    // save the info of this pixel
                    hi.area += 1;
                    sumx += nowx;
                    sumy += nowy;
                    if (nowx < hi.xmin) hi.xmin = nowx;
                    if (nowx > hi.xmax) hi.xmax = nowx;
                    if (nowy < hi.ymin) hi.ymin = nowy;
                    if (nowy > hi.ymax) hi.ymax = nowy;

                    // look around: every pixel within r in x and y
                    for (int dy = -r; dy <= r; dy++){
                        for (int dx = -r; dx <= r; dx++){
                            int newx = nowx + dx;
                            int newy = nowy + dy;
                            if (newx >= 0 && newx < w && newy >= 0 && newy < h){
                                if (mask[newy*w + newx]){
                                    mask[newy*w + newx] = false;
                                    todo.push_back(newy*w + newx);
                                }
                            }
                        }
                    }
                }
                if (hi.area < minarea ) {continue;}  // ignore small blobs
                hi.mx = sumx / hi.area;
                hi.my = sumy / hi.area;
                if (labels) for (int p : members) (*labels)[p] = blobs.size();
                blobs.push_back(hi);
            }
        }
    }
    return blobs;
}
