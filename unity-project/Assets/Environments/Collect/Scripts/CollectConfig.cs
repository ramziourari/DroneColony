using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;

using System.Collections.Generic;

public class CollectConfig : MonoBehaviour
{
    public CollectController exec_drone;
    public CollectDecision dec_drone;
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
    public float startAngle; // each agent will start at different startig angle relative to origine
    public float startRadius;
    public float maxRespawnRadius;
    public float minRespawnRadius; // each agent wil start from a different radius relative to origine

    public List<CollectController> agents;   
    public float arenaID;   
    EnvironmentParameters envParameters;
    public float observationSize;
    public void Awake() 
    {
        Academy.Instance.InferenceSeed = 12345;
        arenaID = Random.Range(0.0f,10000.0f);
        envParameters = Academy.Instance.EnvironmentParameters;
        
        // --- set environment parameters ---
        this.exec_drone.nearestNeighbors = envParameters.GetWithDefault("nearestNeighbors", 2.0f); // how many neighbors to consider always n + 1.
        observationSize = envParameters.GetWithDefault("observationSize", 10.0f); // 7 + 4 x (nearestNeighbors -1)
        var behaviorParams = dec_drone.gameObject.GetComponent<BehaviorParameters>();
        behaviorParams.BrainParameters.VectorObservationSize = (int)observationSize;
    }
    public void Start()
    {        
        numDrones = envParameters.GetWithDefault("num_drones", 8.0f); // how many drones to start with
        minRespawnRadius = envParameters.GetWithDefault("minRespawnRadius", 30.0f); // how far from the center to respawn target and drone
        maxRespawnRadius = envParameters.GetWithDefault("maxRespawnRadius", 30.0f); // how far from the center to respawn target and drone
        
        CreateDrone(numDrones, exec_drone);
    }

    public void CreateDrone(float num, CollectController d_agent)
    {
        startAngle = 0.0f;        
        for (int i = 0; i < num; i++)
        {
            CollectController drone = Instantiate(original: d_agent, parent: this.environment.transform);
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
