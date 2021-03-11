
#ifndef __NETWORK_EVALUATE_H__
#define __NETWORK_EVALUATE_H__
#define maxVelocity 0.1f
#define maxYaw 85.0f

#include <math.h>
/*
 * since the network outputs thrust on each motor,
 * we need to define a struct which stores the values
*/
typedef struct control_t_n {
	float roll; 
	float pitch;
	float yaw;
} control_t_n;

void networkEvaluate(control_t_n *control_n, const float *state_array);
#endif