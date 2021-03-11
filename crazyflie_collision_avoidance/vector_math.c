#include <math.h>

#include "vector_math.h"
#include "stabilizer.h"

void direction(Vector2 *dir, float yaw)
{
        dir -> x = cos(yaw);
        dir -> y = sin(yaw);
}

void bearing(float *b, Vector2 vec1, Vector2 vec2)
{
    float dotProduct = dot(vec1, vec2);
    *b = acos(dotProduct / (magnitude(vec1) * magnitude(vec2)));
    *b /= (float)M_PI;
}

void sub(Vector2 *v, Vector2 vec1, Vector2 vec2)
{
    v->x = vec2.x - vec1.x;
    v->y = vec2.y - vec1.y;
}

float magnitude(Vector2 vec)
{
    return sqrtf(vec.x * vec.x + vec.y * vec.y);
}

void updateVector(Vector2 *vec1, point_t vec2)
{
        vec1 -> x = vec2.x;
        vec1 -> y = vec2.y;
}

float dot(Vector2 vec1, Vector2 vec2)
{
    return vec1.x * vec2.x + vec1.y * vec2.y;
}
