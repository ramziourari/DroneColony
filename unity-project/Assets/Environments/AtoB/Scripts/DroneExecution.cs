using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class DroneExecution : MonoBehaviour
{
    public DroneDecision dec_drone; // the rl agent
    public EnvironmentConfig environment; // this will create n drones and place them in circle

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
    // --- target ---
    public Transform m_target;
    [HideInInspector] public Vector3 m_DirToTarget;
    public float target_x;
    public float target_y;
    public float target_z;

    // --- agent egocentric observations ---
    public Vector3 targetRelativePosition;
    public Vector3 agentPosition;
    public Vector3 agentVelocity;
    public Vector3 agentNormPos;
    public Vector3 agentNormVel;
    public List<DroneExecution> team; // list of active drones
    public List<DroneExecution> nClosest; // list of nearest n drones

    // Start is called before the first frame update
    public void Start()
    {             
        // reset movement + random initial orientation
        droneBody.velocity = new Vector3(0,0,0); 
        droneBody.transform.Rotate(Vector3.up, UnityEngine.Random.Range(0.0f, 360.0f));

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
            DroneExecution m_ag = ag.GetComponent<DroneExecution>();
            if (m_ag.arenaID == this.arenaID) // check on same team
            {
                team.Add(m_ag);
            }
        }
    }

    void FixedUpdate()
    {
        StabilizeDrone();

        m_DirToTarget = this.droneBody.transform.position - this.m_target.transform.position;
        
        // --- agent (6) ---
        agentPosition = this.droneBody.transform.localPosition;
        agentVelocity = this.droneBody.velocity;
        // normalization
        x_pos = agentPosition.x / 73.0f;
        y_pos = agentPosition.y / 20.0f;
        z_pos = agentPosition.z / 73.0f;

        vel_x = (Mathf.Sqrt(2) / max_lv) * agentVelocity.x;
        vel_y = agentVelocity.y / max_lv;
        vel_z = (Mathf.Sqrt(2) / max_lv) * agentVelocity.z;

        // --- target (3)---
        targetRelativePosition = this.transform.localPosition - this.m_target.localPosition;
        // normalization
        target_x = targetRelativePosition.x / (this.environment.ground.localScale.x);
        target_y = targetRelativePosition.y / 20f;
        target_z = targetRelativePosition.z / (this.environment.ground.localScale.z);

        float max2dDistance = Mathf.Sqrt(Mathf.Pow(this.environment.ground.localScale.x, 2) + Mathf.Pow(this.environment.ground.localScale.z, 2));
        float max3dDistance = Mathf.Sqrt(Mathf.Pow(max2dDistance, 2) + Mathf.Pow(20.0f, 2));
        distanceTarget = ((this.transform.localPosition - this.m_target.transform.localPosition).magnitude / max2dDistance);

        // --- execute actions ---
        this.droneBody.velocity += droneBody.transform.forward * this.dec_drone.pitch * max_lv * Time.deltaTime;
        this.droneBody.velocity += droneBody.transform.right * this.dec_drone.roll * max_lv * Time.deltaTime;
        this.droneBody.velocity += droneBody.transform.up * this.dec_drone.thrust * max_lv * Time.deltaTime;
        this.droneBody.angularVelocity += this.droneBody.transform.up * this.dec_drone.yaw * max_av * Time.deltaTime;

        agentNormPos =  new Vector3 (x_pos, y_pos, z_pos);
        agentNormVel = new Vector3 (vel_x, vel_y, vel_z);

        nClosest = team.OrderBy(t=>(t.droneBody.transform.position - this.droneBody.transform.position).sqrMagnitude)
        
                                .Take(4)   // we need 3 but own drone will be counted so always add one :p
        
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
}
