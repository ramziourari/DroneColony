using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;
using System.Collections.ObjectModel;

public class DroneDecision : Agent
{
    public DroneExecution exec_drone;
    public EnvironmentConfig environment;

    // --- agent data ---
    public float perceptionRadius;
    public float minDistanceToAgent;
    public float minDistanceToATarget;
    public float agentPenaltyConst;
    public float episodeLength;
    public float targetConst;
    
    // --- agent actions ---
    public float pitch;
    public float roll;
    public float thrust;
    public float yaw;

    EnvironmentParameters envParameters;

    public override void OnEpisodeBegin()
    {                      
        // --- env parameters to be changed in python ---
        envParameters = Academy.Instance.EnvironmentParameters;
        
        episodeLength = envParameters.GetWithDefault("episodeLength", 5000.0f); // ep. length
        this.exec_drone.max_lv = envParameters.GetWithDefault("max_lv", 20.0f); // max linear velocity
        this.exec_drone.max_av = envParameters.GetWithDefault("max_av", 10.0f); // max angular velocity
        this.perceptionRadius = envParameters.GetWithDefault("perceptionRadius", 0.05f); // agent detection radius (agents only not target)
        minDistanceToAgent = envParameters.GetWithDefault("minDistanceToAgent", 0.03f); // under this distance agent has collided
        minDistanceToATarget = envParameters.GetWithDefault("minDistanceToATarget", 0.1f); // under this distance agent on target
        agentPenaltyConst = envParameters.GetWithDefault("agentPenaltyConst", -10.0f); // under this distance agent on target
        targetConst = envParameters.GetWithDefault("targetConst", -1.0f); // under this distance agent on target
        
        var decisionRequester = gameObject.GetComponent<DecisionRequester>();
        decisionRequester.DecisionPeriod = 1; // DecisionPeriod * Time.fixedDeltaTime = decision period in seconds (default decision after each 20ms)
        decisionRequester.TakeActionsBetweenDecisions = false;
        this.MaxStep = (int)episodeLength;
                
        // respawn agent
        this.exec_drone.Start();
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        // --- egocentric observations ---     
        sensor.AddObservation(this.exec_drone.x_pos);
        // sensor.AddObservation(this.exec_drone.y_pos);
        sensor.AddObservation(this.exec_drone.z_pos);

        sensor.AddObservation(this.exec_drone.vel_x);
        // sensor.AddObservation(this.exec_drone.vel_y);
        sensor.AddObservation(this.exec_drone.vel_z);

        sensor.AddObservation(this.exec_drone.target_x);
        // sensor.AddObservation(this.exec_drone.target_y);
        sensor.AddObservation(this.exec_drone.target_z);
        
        // Monitor.Log("x pos:", (this.exec_drone.x_pos).ToString());
        // Monitor.Log("z pos:", (this.exec_drone.z_pos).ToString());

        // Monitor.Log("x vel:", (this.exec_drone.vel_x).ToString());
        // Monitor.Log("z vel:", (this.exec_drone.vel_z).ToString());

        // Monitor.Log("target x:", (this.exec_drone.target_x).ToString());
        // Monitor.Log("target z:", (this.exec_drone.target_z).ToString());

        // --- peers observations ---
        foreach (DroneExecution ag in this.exec_drone.nClosest)
        {
            float normDistanceToAgent = (1 / (2.0f * Mathf.Sqrt(2)) * (ag.agentNormPos - this.exec_drone.agentNormPos)).magnitude;

            if(ag.gameObject.GetInstanceID() != this.gameObject.GetInstanceID() && (normDistanceToAgent <= perceptionRadius))
            {
                Monitor.Log("norm_distance to agent:", (normDistanceToAgent).ToString(), this.transform);
                var relativeAgentPosition = 0.5f * (ag.agentNormPos - this.exec_drone.agentNormPos);
                var relativeAgentVelocity = 0.5f * (ag.agentNormVel - this.exec_drone.agentNormVel);

                try
                {
                    sensor.AddObservation(relativeAgentPosition.x);
                    // sensor.AddObservation(relativeAgentPosition.y);
                    sensor.AddObservation(relativeAgentPosition.z);

                    sensor.AddObservation(relativeAgentVelocity.x);
                    // sensor.AddObservation(relativeAgentVelocity.y);
                    sensor.AddObservation(relativeAgentVelocity.z);
                }
                catch (Exception)
                {
                    sensor.AddObservation(0.0f);
                    // sensor.AddObservation(0.0f);
                    sensor.AddObservation(0.0f);

                    sensor.AddObservation(0.0f);
                    // sensor.AddObservation(0.0f);
                    sensor.AddObservation(0.0f);
                }

                // Monitor.Log("other_agent_pos_x:", (relativeAgentPosition.x).ToString());
                // // Monitor.Log("other_agent_pos_y:", (relativeAgentPosition.y).ToString());
                // Monitor.Log("other_agent_pos_z:", (relativeAgentPosition.z).ToString());

                // Monitor.Log("other_agent_vel_x:", (relativeAgentVelocity.x).ToString());
                // // Monitor.Log("other_agent_vel_y:", (relativeAgentVelocity.y).ToString());
                // Monitor.Log("other_agent_vel_z:", (relativeAgentVelocity.z).ToString());
            }
            else if (ag.gameObject.GetInstanceID() != this.gameObject.GetInstanceID() && (normDistanceToAgent > perceptionRadius))
            {
                Monitor.Log("norm_distance to agent:", (normDistanceToAgent).ToString(), this.transform);
                sensor.AddObservation(0.0f);
                // sensor.AddObservation(0.0f);
                sensor.AddObservation(0.0f);

                sensor.AddObservation(0.0f);
                // sensor.AddObservation(0.0f);
                sensor.AddObservation(0.0f);
            }
        }
    }

    public override void OnActionReceived(float[] act)
    {  
        // --- infer actions ---  
        pitch = Mathf.Clamp(act[0], -1, 1);
        roll = Mathf.Clamp(act[1], -1, 1);
        yaw = Mathf.Clamp(act[2], -1, 1);
        // thrust = Mathf.Clamp(act[3], -1, 1);

        // --- compute step rewards ---
        RewardTargetFunction();
        RewardAgentsFunction();
        Monitor.Log("n closest neighbours:", (this.exec_drone.nClosest.Count - 1).ToString());          
    }

    public void RewardTargetFunction()
    {
        // Monitor.Log("distance to target:", (this.exec_drone.distanceTarget).ToString());
        AddReward(targetConst * ((minDistanceToATarget - this.exec_drone.distanceTarget)));
    }
    void RewardAgentsFunction()
    {
        foreach (DroneExecution ag in this.exec_drone.nClosest.ToArray())
        {
            if(ag.gameObject.GetInstanceID() != this.gameObject.GetInstanceID())
            {
               float normDistanceToAgent = (1 / (2.0f * Mathf.Sqrt(2)) * (ag.agentNormPos - this.exec_drone.agentNormPos)).magnitude;
                if (normDistanceToAgent < minDistanceToAgent)
                {
                    Monitor.Log("critical distance to other agent:", (normDistanceToAgent).ToString(), this.transform);
                    var relativeAgentPosition = 0.5f * (ag.agentNormPos - this.exec_drone.agentNormPos);
                    var relativeAgentVelocity = 0.5f * (ag.agentNormVel - this.exec_drone.agentNormVel);
                    AddReward(agentPenaltyConst);
                }
            }
        }
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
}
