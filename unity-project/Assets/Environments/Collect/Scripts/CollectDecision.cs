using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;
using System.Collections.ObjectModel;

public class CollectDecision : Agent
{
    public CollectController exec_drone;
    public CollectConfig environment;

    // --- agent data ---
    public float perceptionRadius;
    public float minDistanceToAgent;
    public float minDistanceToATarget;
    public float agentPenaltyConst;
    public float episodeLength;
   
    // --- agent actions ---
    public float pitch;
    public float roll;
    public float thrust;
    public float yaw;
    public float msg;
    // --- sensor noise ---
    float linearVelocityStd = 0.01f;
    float bearingStd = 0.001f;
    float distanceStd = 0.0001f;
    float positionStd = 0.001f;

    EnvironmentParameters envParameters;

    // public List<float> distanceBins = new List<float> {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    // float[] distanceFreq = new float [10];

    // public List<float> bearingBins = new List<float> {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    // float[] bearingFreq = new float [10];
    // void FixedUpdate() 
    // {
    //     foreach (CollectController ag in this.exec_drone.team)
    //     {
    //         Array.Clear(distanceFreq, 0, distanceFreq.Length);            
    //         float d = 0.0f;
    //         for(int i=0; i < distanceBins.Count; i++)
    //         {
    //             if (distanceBins[i] > d)
    //             {
    //                 distanceFreq[i] += 1;
    //                 break;
    //             }
    //         }

    //         Array.Clear(bearingFreq, 0, bearingFreq.Length);            
    //         float b = 0.0f;
    //         for(int i=0; i < bearingBins.Count; i++)
    //         {
    //             if (bearingBins[i] > b)
    //             {
    //                 bearingFreq[i] += 1;
    //                 break;
    //             }
    //         }
    //     }

    // }
    public override void OnEpisodeBegin()
    {                      
        // --- env parameters to be changed in python ---
        envParameters = Academy.Instance.EnvironmentParameters;        
        episodeLength = envParameters.GetWithDefault("episodeLength", 5000.0f); // ep. length
        this.exec_drone.max_lv = envParameters.GetWithDefault("max_lv", 20.0f); // max linear velocity
        this.exec_drone.max_av = envParameters.GetWithDefault("max_av", 10.0f); // max angular velocity
        this.perceptionRadius = envParameters.GetWithDefault("perceptionRadius", 10.0f); // agent detection radius (agents only not target)
        minDistanceToAgent = envParameters.GetWithDefault("minDistanceToAgent", 7f); // under this distance agent has collided
        minDistanceToATarget = envParameters.GetWithDefault("minDistanceToATaAddObservationrget", 3.5f); // under this distance agent on target
        agentPenaltyConst = envParameters.GetWithDefault("agentPenaltyConst", -1.0f); // under this distance agent on target
        
        var decisionRequester = gameObject.GetComponent<DecisionRequester>();
        decisionRequester.DecisionPeriod = 2;
        this.exec_drone.Start();
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        // // --- egocentric observations ---     
        // sensor.AddObservation(this.exec_drone.x_pos);
        // sensor.AddObservation(this.exec_drone.z_pos);

        // sensor.AddObservation(this.exec_drone.vel_x);
        // sensor.AddObservation(this.exec_drone.vel_z);
        float velocityNoise = NextGaussian(0.0f, linearVelocityStd, -3.0f * linearVelocityStd, 3.0f * linearVelocityStd);
        float bearingNoise = NextGaussian(0.0f, bearingStd, -3.0f * bearingStd, 3.0f * bearingStd);
        float distanceNoise = NextGaussian(0.0f, distanceStd, -3.0f * distanceStd, 3.0f * distanceStd);
        float positionNoise = NextGaussian(0.0f, positionStd, -3.0f * positionStd, 3.0f * positionStd);
        // sensor.AddObservation(this.exec_drone.target_x);
        // sensor.AddObservation(this.exec_drone.target_z);

        // Monitor.Log("n closest neighbours:", (this.exec_drone.nClosest.Count - 1).ToString());          
        
        float bearingToTarget = this.exec_drone.targetBearing;
        bearingToTarget /= 180.0f; // [0, 1]
        // Monitor.Log("bearing to target:", (bearingToTarget / 180.0f).ToString(), this.transform);   // [-1, 1]    

        float distanceToTarget = (this.exec_drone.m_DirToTarget.magnitude) / 212.0f;
        // Monitor.Log("distance to target normalized:", (distanceToTarget).ToString()); // [0, 1]
        
        float agentPos_x = this.exec_drone.droneBody.transform.localPosition.x/73.0f;
        float agentPos_y = this.exec_drone.droneBody.transform.localPosition.y/20.0f;
        float agentPos_z = this.exec_drone.droneBody.transform.localPosition.z/73.0f;
        // Monitor.Log("agent pos x:", (agentPos_x).ToString()); // [-1, 1] 
        // Monitor.Log("agent pos y:", (agentPos_y).ToString());          
        // Monitor.Log("agent pos z:", (agentPos_y).ToString());   

        float agentVelocity_x = this.exec_drone.vel_x;
        float agentVelocity_y = this.exec_drone.vel_y;
        float agentVelocity_z = this.exec_drone.vel_z;
        // Monitor.Log("agent velocity x:", (agentVelocity_x).ToString());   // [-1, 1]
        // Monitor.Log("agent velocity y:", (agentVelocity_y).ToString());
        // Monitor.Log("agent velocity z:", (agentVelocity_z).ToString());
        
        //  --- egocentric observations ---
        // sensor.AddObservation(distanceToTarget + distanceNoise);
        sensor.AddObservation(distanceToTarget);
        // Monitor.Log("distance to target:", (distanceToTarget).ToString());          
        // Monitor.Log("distanceToTarget:", (distanceToTarget).ToString(), this.transform); // [0, 1]
        // sensor.AddObservation(bearingToTarget + bearingNoise);
        sensor.AddObservation(bearingToTarget);
        // Monitor.Log("distance to target:", (distanceToTarget).ToString());          

        // sensor.AddObservation(agentPos_x + positionNoise);
        // sensor.AddObservation(agentPos_z + positionNoise);

        sensor.AddObservation(agentPos_x);
        sensor.AddObservation(agentPos_z);
        
        // sensor.AddObservation(agentVelocity_x + velocityNoise);
        // sensor.AddObservation(agentVelocity_z + velocityNoise);
        sensor.AddObservation(agentVelocity_x);
        sensor.AddObservation(agentVelocity_z);
        // sensor.AddObservation(msg);
        
        // --- peers observations ---
        foreach (CollectController ag in this.exec_drone.nClosest)
        {
            // float normDistanceToAgent = (1 / (2.0f * Mathf.Sqrt(2)) * (ag.agentNormPos - this.exec_drone.agentNormPos)).magnitude;
            
            // distance to other agent [0, 1]
            var dirToAgent = ag.droneBody.transform.localPosition - this.exec_drone.droneBody.transform.localPosition;
            float distanceToAgent = dirToAgent.magnitude;
            distanceToAgent /= 212.0f;
            
            // brearing to other agent [0, 1]
            var agentCos = Vector3.Dot(this.exec_drone.droneBody.transform.forward, dirToAgent) / dirToAgent.magnitude;
            float bearingToAgent = Mathf.Acos(agentCos) * Mathf.Rad2Deg;
            bearingToAgent /= 180.0f;

            if(ag.gameObject.GetInstanceID() != this.gameObject.GetInstanceID() && (distanceToAgent <= perceptionRadius))
            {
                // Monitor.Log("norm_distance to agent:", (distanceToAgent).ToString());
                var relativeAgentPosition = 0.5f * (ag.agentNormPos - this.exec_drone.agentNormPos);
                var relativeAgentVelocity = 0.5f * (ag.agentNormVel - this.exec_drone.agentNormVel);
                // Monitor.Log("bearing to Agent:", (bearingToAgent).ToString(), this.transform); // [0, 1]
                try
                {
                    // sensor.AddObservation(relativeAgentPosition.x);
                    // sensor.AddObservation(relativeAgentPosition.z);

                    // sensor.AddObservation(distanceToAgent + distanceNoise);
                    // // Monitor.Log("norm distance to agent:", (distanceToAgent).ToString());
                    // sensor.AddObservation(bearingToAgent + bearingNoise);

                    sensor.AddObservation(distanceToAgent);
                    // Monitor.Log("norm distance to agent:", (distanceToAgent).ToString());
                    sensor.AddObservation(bearingToAgent);

                    // sensor.AddObservation(relativeAgentVelocity.x + 2 * velocityNoise);
                    // sensor.AddObservation(relativeAgentVelocity.z + 2 * velocityNoise);

                    sensor.AddObservation(relativeAgentVelocity.x);
                    sensor.AddObservation(relativeAgentVelocity.z);
                }
                catch (Exception)
                {
                    sensor.AddObservation(0.0f);
                    sensor.AddObservation(0.0f);

                    // sensor.AddObservation(0.0f);

                    sensor.AddObservation(0.0f);
                    sensor.AddObservation(0.0f);
                }
            }
            else if (ag.gameObject.GetInstanceID() != this.gameObject.GetInstanceID() && (distanceToAgent > perceptionRadius))
            {
                sensor.AddObservation(0.0f);
                sensor.AddObservation(0.0f);

                // sensor.AddObservation(0.0f);

                sensor.AddObservation(0.0f);
                sensor.AddObservation(0.0f);
            }
            // if(ag.gameObject.GetInstanceID() != this.gameObject.GetInstanceID() && (distanceToAgent <= perceptionRadius))
            // {
            //     // Monitor.Log("norm_distance to agent:", (distanceToAgent).ToString());
            //     var relativeAgentPosition = 0.5f * (ag.agentNormPos - this.exec_drone.agentNormPos);
            //     var relativeAgentVelocity = 0.5f * (ag.agentNormVel - this.exec_drone.agentNormVel);

            //     try
            //     {
            //         sensor.AddObservation(relativeAgentPosition.x);
            //         sensor.AddObservation(relativeAgentPosition.z);

            //         sensor.AddObservation(relativeAgentVelocity.x);
            //         sensor.AddObservation(relativeAgentVelocity.z);
            //     }
            //     catch (Exception)
            //     {
            //         sensor.AddObservation(0.0f);
            //         sensor.AddObservation(0.0f);

            //         sensor.AddObservation(0.0f);
            //         sensor.AddObservation(0.0f);
            //     }
            // }
            // else if (ag.gameObject.GetInstanceID() != this.gameObject.GetInstanceID() && (distanceToAgent > perceptionRadius))
            // {
            //     sensor.AddObservation(0.0f);
            //     sensor.AddObservation(0.0f);

            //     sensor.AddObservation(0.0f);
            //     sensor.AddObservation(0.0f);
            // }
        }
    }

    public override void OnActionReceived(float[] act)
    {  
        // --- infer actions ---  
        pitch = Mathf.Clamp(act[0], -1, 1);
        roll = Mathf.Clamp(act[1], -1, 1);
        yaw = Mathf.Clamp(act[2], -1, 1);
        // msg = Mathf.Clamp(act[3], -1, 1);

        // --- compute step rewards ---
        RewardAgentsFunction();
        RewardFunctionMovingTowards();
        AddReward(-0.001f);
        //  check if on target if true respawn target randomly
        float distanceToTarget = (this.exec_drone.m_DirToTarget.magnitude);
        if (distanceToTarget < minDistanceToATarget)
        {   
            // AddReward(1.0f);
            this.exec_drone.respawnTarget();
        }
    }
    void RewardAgentsFunction()
    {
        foreach (CollectController ag in this.exec_drone.nClosest.ToArray())
        {
            if(ag.gameObject.GetInstanceID() != this.gameObject.GetInstanceID())
            {
                var dirToAgent = ag.droneBody.transform.localPosition - this.exec_drone.droneBody.transform.localPosition;
                float distanceToAgent = dirToAgent.magnitude;                
                if (distanceToAgent < minDistanceToAgent)
                {
                    // Monitor.Log("penalty distance:", (distanceToAgent).ToString());
                    AddReward(agentPenaltyConst);
                }
            }
        }
    }
    void RewardFunctionMovingTowards()
    {
        float m_MovingTowardsDot = Vector3.Dot(this.exec_drone.targetRelativePosition.normalized, this.exec_drone.droneBody.velocity);
        // Monitor.Log("m_MovingTowardsDot:", (m_MovingTowardsDot).ToString(), this.transform);
        AddReward(0.003f * m_MovingTowardsDot);
    }
    public override void Heuristic(float[] actionsOut)
    {
		if(Input.GetKey(KeyCode.A)) {this.exec_drone.droneBody.velocity += this.exec_drone.droneBody.transform.right * -this.exec_drone.max_lv  * Time.deltaTime;}
		if(Input.GetKey(KeyCode.D)) {this.exec_drone.droneBody.velocity += this.exec_drone.droneBody.transform.right * this.exec_drone.max_lv  * Time.deltaTime;}
		if(Input.GetKey(KeyCode.I)) {this.exec_drone.droneBody.velocity += this.exec_drone.droneBody.transform.up * this.exec_drone.max_lv * Time.deltaTime;}
		if(Input.GetKey(KeyCode.K)) {this.exec_drone.droneBody.velocity += this.exec_drone.droneBody.transform.up * -this.exec_drone.max_lv * Time.deltaTime;}
		if(Input.GetKey(KeyCode.W)) {this.exec_drone.droneBody.velocity += this.exec_drone.droneBody.transform.forward * this.exec_drone.max_lv  * Time.deltaTime;}
        if(Input.GetKey(KeyCode.S)) {this.exec_drone.droneBody.velocity += this.exec_drone.droneBody.transform.forward * -this.exec_drone.max_lv * Time.deltaTime;}
		if(Input.GetKey(KeyCode.J)) {this.exec_drone.droneBody.angularVelocity += this.exec_drone.droneBody.transform.up * -this.exec_drone.max_av * Time.deltaTime;}
		if(Input.GetKey(KeyCode.L)) {this.exec_drone.droneBody.angularVelocity += this.exec_drone.droneBody.transform.up * this.exec_drone.max_av * Time.deltaTime;}
    }	
    public static float NextGaussian() 
    {
        float v1, v2, s;
        do {
            v1 = 2.0f * UnityEngine.Random.Range(0f,1f) - 1.0f;
            v2 = 2.0f * UnityEngine.Random.Range(0f,1f) - 1.0f;
            s = v1 * v1 + v2 * v2;
        } while (s >= 1.0f || s == 0f);
    
        s = Mathf.Sqrt((-2.0f * Mathf.Log(s)) / s);
    
        return v1 * s;
    }    	
    public static float NextGaussian(float mean, float standard_deviation)
    {
        return mean + NextGaussian() * standard_deviation;
    }
    public float NextGaussian(float mean, float standard_deviation, float min, float max) 
    {
        float x;
        do {
        x = NextGaussian(mean, standard_deviation);
        } while (x < min || x > max);
        return x;
    }
}