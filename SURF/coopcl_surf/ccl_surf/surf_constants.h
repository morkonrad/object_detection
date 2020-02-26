#pragma once

#define DESC_SIZE 64
#define DESC_TH_WGS 81
#define DESC_WGS_ITEM 16
#define ORIENT_WGS_STEP1 169
#define ORIENT_WGS_STEP2 42

static const int OCTAVES = 3;
static const int MAX_OCTAVES = 5;
static const int INTERVALS = 3;
static const int MAX_INTERVALS = 4;
static const float THRES = 0.0001f;
static const int SAMPLE_STEP = 5;
static const int INIT_FEATS = 8000;