#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "stabilizer.h"

typedef struct _Vector2
{
  float x;
  float y;
} Vector2;

void direction(Vector2 *dir, float yaw);
void bearing(float *b, Vector2 vec1, Vector2 vec2);
void sub(Vector2 *v,  Vector2 vec1, Vector2 vec2);
float magnitude(Vector2 vec);
void updateVector(Vector2 *vec1, point_t vec2);
float dot(Vector2 vec1, Vector2 vec2);