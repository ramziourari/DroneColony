using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;

using System.Collections.Generic;

public class EnvironmentConfig : MonoBehaviour
{
    public DroneExecution exec_drone;
    public DroneDecision dec_drone;
    public Transform target;
    public Transform environment;
    
    // Arena boundaries
    public Transform ground;
    public Transform roof;
    public Transform wall_north;
    public Transform wall_south;
    public Transform wall_east;
    public Transform wall_west;

    public float numDrones; // how many drones to create
    public float addDrone;
    public float startAngle; // each agent will start at different startig angle relative to origine
    public float startRadius; // each agent wil start from a different radius relative to origine
    public float maxRespawnRadius;
    public float minRespawnRadius; // each agent wil start from a different radius relative to origine    

    public List<DroneExecution> agents;
    public List<DroneExecution> activeAgents;

    public float arenaID;   
    EnvironmentParameters envParameters;
    public float observationSize;

    public void Awake() 
    {
        Academy.Instance.InferenceSeed = 12345;
        arenaID = Random.Range(0.0f,10000.0f);        
        envParameters = Academy.Instance.EnvironmentParameters;
    }
    public void Start()
    {
        observationSize = envParameters.GetWithDefault("observationSize", 10.0f);
        numDrones = envParameters.GetWithDefault("num_drones", 2.0f); // how many drones to start with
        minRespawnRadius = envParameters.GetWithDefault("minRespawnRadius", 12.0f); // how far from the center to respawn target and drone min
        maxRespawnRadius = envParameters.GetWithDefault("maxRespawnRadius", 13.0f); // how far from the center to respawn target and drone max
        var behaviorParams = dec_drone.gameObject.GetComponent<BehaviorParameters>();
        behaviorParams.BrainParameters.VectorObservationSize = (int)observationSize;
        
        CreateDrone(numDrones, exec_drone);
    }

    public void CreateDrone(float num, DroneExecution d_agent)
    {
        startAngle = 0.0f;        
        for (int i = 0; i < num; i++)
        {
            DroneExecution drone = Instantiate(original: d_agent, parent: this.environment.transform);
            Transform _target = Instantiate(original: target, parent: this.environment.transform);

            startAngle = (2 * Mathf.PI/this.numDrones) * i;
            startRadius = UnityEngine.Random.Range(minRespawnRadius, maxRespawnRadius);

            drone.m_target = _target;
            drone.respawnAngle = startAngle;
            drone.respawnRadius = startRadius;
            drone.arenaID = arenaID;
            
            agents.Add(drone);
        }
    }
}
