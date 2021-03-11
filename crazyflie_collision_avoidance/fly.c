#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


#include "app.h"
#include "commander.h"
#include "FreeRTOS.h"
#include "task.h"
#include "debug.h"
#include "log.h"
#include "estimator_kalman.h"
#include "radiolink.h"
#include "stabilizer.h"
#include "vector_math.h"
#include "nn.h"

#define DEBUG_MODULE "PUSH"
#define maxX 1.4f
#define maxY 2.0f
#define maxDistance 4.88262224629f // sqrt(2,8² + 4²) = 4.88262224629 m

typedef struct _PacketData
{
  int id;
  Vector2 pos;
  Vector2 vel;
} PacketData;  // size: ? bytes + 12 bytes + 12 bytes  // max: 60 bytes

static PacketData packetData;  // the data that is send around via the p2p broadcast method
static setpoint_t setpoint;
static point_t kalmanPosition;
static point_t kalmanVelocity;
// -- agent own data --
Vector2 target = {-0.5f, 0.0f}; // target x,y in world frame
static float yaw;
float bearingToTarget;
float distanceToTarget;
static Vector2 lookDirection;
static Vector2 ownPosition;
static Vector2 ownVelocity;
static Vector2 dirToTarget;
// -- agent peers data --
PacketData receivedPacketData;
static Vector2 dirToAgent;
static float distanceToNearestAgent = 9999.0f;
Vector2 otherPosition;
Vector2 otherVelocity;
Vector2 relativeVelocity;
float bearingToAgent;
// -- swarm config --
int numDrones = 4;
// -- network specific --
static float state_array[10];
static control_t_n control_n;
static int id = 1;

// Stock log groups
LOG_GROUP_START(fly)
  LOG_ADD(LOG_FLOAT, fk_pos_x, &kalmanPosition.x)
  LOG_ADD(LOG_FLOAT, fk_pos_y, &kalmanPosition.y)
  LOG_ADD(LOG_FLOAT, lookDir_x, &lookDirection.x)
  LOG_ADD(LOG_FLOAT, lookDir_y, &lookDirection.y)
  LOG_ADD(LOG_FLOAT, d_yaw, &yaw)
  LOG_ADD(LOG_FLOAT, bToTarget, &bearingToTarget)
  LOG_ADD(LOG_FLOAT, dToTarget, &distanceToTarget)
LOG_GROUP_STOP(fly)

LOG_GROUP_START(packet)
  LOG_ADD(LOG_INT32, ID, &packetData.id)
  LOG_ADD(LOG_FLOAT, POS, &packetData.pos)
  LOG_ADD(LOG_FLOAT, VEL, &packetData.vel)
LOG_GROUP_STOP(fly)

LOG_GROUP_START(nn)
  LOG_ADD(LOG_FLOAT, k_pos_x, &kalmanPosition.x)
  LOG_ADD(LOG_FLOAT, k_pos_y, &kalmanPosition.y)
  LOG_ADD(LOG_FLOAT, pos_x, &state_array[0])
  LOG_ADD(LOG_FLOAT, pos_y, &state_array[1])
  LOG_ADD(LOG_FLOAT, vel_x, &state_array[2])
  LOG_ADD(LOG_FLOAT, vel_y, &state_array[3])
  LOG_ADD(LOG_FLOAT, dToTarget, &state_array[4])
  LOG_ADD(LOG_FLOAT, bToTarget, &state_array[5])
  LOG_ADD(LOG_FLOAT, pitch, &control_n.pitch)
  LOG_ADD(LOG_FLOAT, roll, &control_n.roll)
  LOG_ADD(LOG_FLOAT, _yaw, &control_n.yaw)
  LOG_ADD(LOG_FLOAT, distaneAgent, &state_array[6])
  LOG_ADD(LOG_FLOAT, bToAgent, &state_array[7])
  LOG_ADD(LOG_FLOAT, relVelX, &state_array[8])
  LOG_ADD(LOG_FLOAT, relVelY, &state_array[9])
LOG_GROUP_STOP(nn)


static void setHoverSetpoint(setpoint_t *setpoint, float vx, float vy, float z, float yawrate)
{
  setpoint->mode.z = modeAbs;
  setpoint->position.z = z;
  

  setpoint->mode.yaw = modeVelocity;
  setpoint->attitudeRate.yaw = yawrate;


  setpoint->mode.x = modeVelocity;
  setpoint->mode.y = modeVelocity;
  setpoint->velocity.x = vx;
  setpoint->velocity.y = vy;

  setpoint->velocity_body = true;
}

static void communicate(int i)
{
    if (i == id)
    {
      P2PPacket packet;
      packet.port = 0;
      packet.size = sizeof(PacketData);
      memcpy(packet.data, &packetData, sizeof(PacketData));
      radiolinkSendP2PPacketBroadcast(&packet);
    }
}

void p2pCallbackHandler(P2PPacket *p)
{
  PacketData receivedPacketData;
  memcpy(&receivedPacketData, p->data, sizeof(PacketData));
  otherPosition = receivedPacketData.pos;
  otherVelocity = receivedPacketData.vel;
}

void updatePacket(PacketData *pack, point_t kalP, point_t kalV)
{
  pack->pos.x = kalP.x;
  pack->pos.y = kalP.y;

  pack->vel.x = kalV.x;
  pack->vel.y = kalV.y;
}

void updateOwn()
{
      estimatorKalmanGetEstimatedPos(&kalmanPosition); // estimate position in world frame
      estimatorKalmanGetEstimatedVelocity(&kalmanVelocity); // estimate velocity in body frame
      
      updatePacket(&packetData, kalmanPosition, kalmanVelocity);
      
      updateVector(&ownPosition, kalmanPosition); // writes kalman position estimations to Vector2
      updateVector(&ownVelocity, kalmanVelocity); // writes kalman velocity estimations to Vector2
      
      getYaw(&yaw); // yaw in radians      
      
      direction(&lookDirection, yaw); // look direction vector of this drone (x: cos(yaw), y: sin(yaw))  
      
      sub(&dirToTarget, ownPosition, target); // vector pointing from drone to target
      
      bearing(&bearingToTarget, dirToTarget, lookDirection); // [0, 1] 0: looking to target, 1: target behind
      
      distanceToTarget = magnitude(dirToTarget); // [0, 1]

      // -- egocentric observations (6) --
      state_array[0] = distanceToTarget / maxDistance; // [0, 1]
      state_array[1] = bearingToTarget; // [0, 1]
      state_array[3] = ownPosition.x / maxX; // [-1, 1]
      state_array[2] = ownPosition.y / maxY; // [-1, 1]
      state_array[5] = ownVelocity.x / maxVelocity; // [-1, 1]
      state_array[4] = ownVelocity.y / maxVelocity; // [-1, 1]
}

void updatePeers()
{
  sub(&dirToAgent, ownPosition, otherPosition);
  if (magnitude(dirToAgent) <= distanceToNearestAgent)
  {  
    distanceToNearestAgent = magnitude(dirToAgent);
    bearing(&bearingToAgent, dirToAgent, lookDirection);
    sub(&relativeVelocity, ownVelocity, otherVelocity);
    
    // -- peers observations (4) --
    state_array[6] = distanceToNearestAgent / (maxDistance + 1.0f);
    state_array[7] = bearingToAgent;
    state_array[9] = relativeVelocity.x / (2 * maxVelocity);
    state_array[8] = relativeVelocity.y / (2 * maxVelocity);
  }
}

void appMain()
{
  vTaskDelay(M2T(5000));
  estimatorKalmanGetEstimatedPos(&kalmanPosition);
  setHoverSetpoint(&setpoint,0.0f, 0.0f, 0.4f, 0.0f);
  commanderSetSetpoint(&setpoint, 3);
  updateOwn();
  p2pRegisterCB(p2pCallbackHandler);

  while(1) 
  {

    vTaskDelay(M2T(10));
    distanceToNearestAgent = 999.0f;
    
    for (int i = 0; i < numDrones; i++)
    {
      updateOwn();
      communicate(i);
      updatePeers();
      vTaskDelay(M2T(10));
    }        
    if (distanceToTarget < 0.03f)
    {
        target.x *= -1.0f;
    }
    networkEvaluate(&control_n, state_array); // output -> roll, pitch, yaw
    setHoverSetpoint(&setpoint, control_n.roll, control_n.pitch, 0.4f, control_n.yaw);
    
    commanderSetSetpoint(&setpoint, 3);
  }
}


