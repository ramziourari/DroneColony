using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class CollectController : MonoBehaviour
{
    public CollectDecision dec_drone; // the rl agent
    public CollectConfig environment; // this will create n drones and place them in circle

    // --- drone data ---
    public Rigidbody droneBody;  // the body {velocity, mass..}
    public float max_lv; // max linear velocity
    public float max_av; // max angular velocity
    public float pitch;
    public float roll;
    public float thrust;
    public float yaw;
    public float x_pos;
    public float y_pos;
    public float z_pos;
    public float vel_x;
    public float vel_y;
    public float vel_z;
    public float distanceTarget;

    // --- initialization parameters ---
    public float respawnRadius;
    public float respawnAngle;
    public float droneMass;
    public float startAngle;
    public float arenaID;
    public float nearestNeighbors;

    // --- target ---
    public Transform m_target;
    [HideInInspector] public Vector3 m_DirToTarget;
    public float target_x;
    public float target_y;
    public float target_z;
    public float targetBearing;

    // --- agent egocentric observations ---
    public Vector3 targetRelativePosition;
    public Vector3 agentPosition;
    public Vector3 agentVelocity;
    public Vector3 agentNormPos;
    public Vector3 agentNormVel;
    public List<CollectController> team; // list of active drones
    public List<CollectController> nClosest; // list of nearest n drones
    GameObject go;
    LineRenderer lr;

    public void Awake() 
    {
        go = new GameObject();
        lr = go.AddComponent<LineRenderer>();
    }
    // Start is called before the first frame update
    public void Start()
    {             
        Time.fixedDeltaTime = 0.02f; // 50 Hz or decision every 20ms
        // reset movement + random initial orientation
        droneBody.velocity = new Vector3(0,0,0); 
        droneBody.transform.Rotate(Vector3.up, UnityEngine.Random.Range(0.0f, 360.0f));
        // float randomAngle = UnityEngine.Random.Range(-2.0f, 2.0f);

        // drone random initial position           
        var newDronetPos = new Vector3(
            respawnRadius * Mathf.Cos(respawnAngle),
            5.0f,
            respawnRadius * Mathf.Sin(respawnAngle));
        // target random initial position
        var newTargetPos = new Vector3(
            respawnRadius * Mathf.Cos(respawnAngle + Mathf.PI),
            5.0f,
            respawnRadius * Mathf.Sin(respawnAngle + Mathf.PI));
        newTargetPos.y = 5.0f;

        droneBody.transform.localPosition = newDronetPos;
        m_target.transform.localPosition = newTargetPos;
        
        // save present drones in team []
        team.Clear();
        GameObject [] activeDrones = GameObject.FindGameObjectsWithTag("agent");
        foreach (GameObject ag in activeDrones)
        {
            CollectController m_ag = ag.GetComponent<CollectController>();
            if (m_ag.arenaID == this.arenaID) // check on same team
            {
                team.Add(m_ag);
            }
        }
    }
    public void respawnTarget() // respawns the target randomly
    {
        var angleOffset = UnityEngine.Random.Range(0, 2 * Mathf.PI);
        var newTargetPos = new Vector3(
            respawnRadius * Mathf.Cos(angleOffset),
            5.0f,
            respawnRadius * Mathf.Sin(angleOffset));
        
        newTargetPos.y = 5.0f;
        m_target.transform.localPosition = newTargetPos;
    }
    void FixedUpdate()
    {
        StabilizeDrone();
        // randomForce();

        m_DirToTarget = this.droneBody.transform.position - this.m_target.transform.position;
        
        // --- agent (6) ---
        agentPosition = this.transform.localPosition;
        agentVelocity = this.droneBody.velocity;
        // normalization
        x_pos = agentPosition.x / this.environment.ground.localScale.x;
        y_pos = agentPosition.y / 20.0f;
        z_pos = agentPosition.z / this.environment.ground.localScale.z;

        vel_x = (Mathf.Sqrt(2) / max_lv) * agentVelocity.x;
        vel_y = agentVelocity.y / max_lv;
        vel_z = (Mathf.Sqrt(2) / max_lv) * agentVelocity.z;

        // --- target (3)---
        targetRelativePosition = this.m_target.localPosition - this.transform.localPosition;
        // normalization
        target_x = targetRelativePosition.x / this.environment.ground.localScale.x;
        target_y = targetRelativePosition.y / 20f;
        target_z = targetRelativePosition.z / this.environment.ground.localScale.z;

        var targetCos = Vector3.Dot(-droneBody.transform.forward, m_DirToTarget) / m_DirToTarget.magnitude;
        targetBearing = Mathf.Acos(targetCos) * Mathf.Rad2Deg;

        float max2dDistance = Mathf.Sqrt(Mathf.Pow(this.environment.ground.localScale.x, 2) + Mathf.Pow(this.environment.ground.localScale.z, 2));
        float max3dDistance = Mathf.Sqrt(Mathf.Pow(max2dDistance, 2) + Mathf.Pow(20.0f, 2));
        distanceTarget = 2 * ((this.transform.localPosition - this.m_target.transform.localPosition).magnitude / max2dDistance);

        // --- execute actions ---
        this.droneBody.velocity += droneBody.transform.forward * this.dec_drone.pitch * max_lv * Time.deltaTime;
        this.droneBody.velocity += droneBody.transform.right * this.dec_drone.roll * max_lv * Time.deltaTime;
        this.droneBody.velocity += droneBody.transform.up * this.dec_drone.thrust * max_lv * Time.deltaTime;
        this.droneBody.angularVelocity += this.droneBody.transform.up * this.dec_drone.yaw * max_av * Time.deltaTime;

        agentNormPos =  new Vector3 (x_pos, y_pos, z_pos);
        agentNormVel = new Vector3 (vel_x, vel_y, vel_z);
    
        nClosest = team.OrderBy(t=>(t.droneBody.transform.position - this.droneBody.transform.position).sqrMagnitude)
    
                            .Take((int)nearestNeighbors)   // own drone will be counted so always give n + 1 :p
    
                            .ToList();       

    }

    void StabilizeDrone()
    {
        // limit speed of drone
        if (this.droneBody.velocity.magnitude >= max_lv)
        {
            this.droneBody.velocity *= 0.99f;
        }
        if (this.droneBody.angularVelocity.magnitude >= max_av)
        {
            this.droneBody.angularVelocity *= 0.99f;
        }

        Vector3 DroneRotation = droneBody.transform.localEulerAngles;

		this.droneBody.AddForce(0,9.80675f * droneBody.mass,0);//drone not lose height very fast, if you want not to lose height al all change 9 into 9.80665
        // stabilize z-axis:
		// big tilt
		if(DroneRotation.z>10 && DroneRotation.z<=180){droneBody.AddRelativeTorque (0, 0, -10);}//if tilt too big(stabilizes drone on z-axis)
		if(DroneRotation.z>180 && DroneRotation.z<=350){droneBody.AddRelativeTorque (0, 0, 10);}//if tilt too big(stabilizes drone on z-axis)
		// small tilt
		if(DroneRotation.z>1 && DroneRotation.z<=10){droneBody.AddRelativeTorque (0, 0, -3);}//if tilt not very big(stabilizes drone on z-axis)
		if(DroneRotation.z>350 && DroneRotation.z<359){droneBody.AddRelativeTorque (0, 0, 3);}//if tilt not very big(stabilizes drone on z-axis)
		// stabilize x-axis:
		//big tilt
		if(DroneRotation.x>10 && DroneRotation.x<=180){droneBody.AddRelativeTorque (-10, 0, 0);}//if tilt too big(stabilizes drone on x-axis)
		if(DroneRotation.x>180 && DroneRotation.x<=350){droneBody.AddRelativeTorque (10, 0, 0);}//if tilt too big(stabilizes drone on x-axis)
		//small tilt
		if(DroneRotation.x>1 && DroneRotation.x<=10){droneBody.AddRelativeTorque (-3, 0, 0);}//if tilt not very big(stabilizes drone on x-axis)
		if(DroneRotation.x>350 && DroneRotation.x<359){droneBody.AddRelativeTorque (3, 0, 0);}//if tilt not very big(stabilizes drone on x-axis)
    }
    void Update() 
    {
        drawLine();
    }

    private void drawLine()
    {     

        lr.SetPosition(0, droneBody.transform.position);
        lr.SetPosition(1, this.m_target.transform.position);
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
    void randomForce()
    {
        this.droneBody.AddForce(
            NextGaussian(mean:0.0f, standard_deviation: 10.0f),
            0,
            NextGaussian(mean:0.0f, standard_deviation:10.0f));
    }

}
