/* ==============================================================================
 * Edge AI Compression Pipeline - Auto-Generated Class Label Mappings
 * Target: 12-Class Keyword Spotting (10 target words + silence + unknown)
 * Generated: 2026-10-17T13:37:58.895676+00:00
 * ============================================================================== */

#ifndef MODEL_LABELS_H_
#define MODEL_LABELS_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MODEL_NUM_CLASSES 12

typedef enum {
    LABEL_DOWN = 0,
    LABEL_GO = 1,
    LABEL_LEFT = 2,
    LABEL_NO = 3,
    LABEL_OFF = 4,
    LABEL_ON = 5,
    LABEL_RIGHT = 6,
    LABEL_STOP = 7,
    LABEL_UP = 8,
    LABEL_YES = 9,
    LABEL_SPECIAL_SILENCE = 10,
    LABEL_SPECIAL_UNKNOWN = 11
} ModelClassId;

extern const char* const kModelLabels[MODEL_NUM_CLASSES];

#ifdef __cplusplus
}
#endif

#if defined(MODEL_LABELS_IMPLEMENTATION) || !defined(MODEL_LABELS_HEADER_ONLY)
const char* const kModelLabels[MODEL_NUM_CLASSES] = {
    "down",
    "go",
    "left",
    "no",
    "off",
    "on",
    "right",
    "stop",
    "up",
    "yes",
    "_silence_",
    "_unknown_"
};
#endif // MODEL_LABELS_IMPLEMENTATION

#endif // MODEL_LABELS_H_
