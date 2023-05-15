using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using RosMessageTypes.Std;
using System.IO;
using System.Threading;
using System.Net;
using System.Net.Sockets;

public class ImagePublisher : MonoBehaviour
{
    ROSConnection ros;
    public Camera imageCamera;
    public string topicName = "/image_topic";
    private ImageMsg imageMessage;
    private int width;
    private int height;
    
    public float interval = 0.1f; // interval in seconds
    private float timeLeft;
    
    WaitForEndOfFrame frameEnd = new WaitForEndOfFrame();

    
    // private Socket socket;
    
    // Start is called before the first frame update
    void Start()
    {
    	width = imageCamera.pixelWidth;
    	height = imageCamera.pixelHeight;
    	
        // start the ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>(topicName);
        
        //socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        //socket.Connect("localhost", 8000); // replace with your server's IP address and port number
        
        
    }

    // Update is called once per frame
    void Update()
    {
	// Your code to run X times per second goes here
	//PublishImage();
	StartCoroutine(PublishImage()); 
    }
    
    
    
    private IEnumerator PublishImage() 
    {
    	yield return new WaitForEndOfFrame();
    	
    	Texture2D img = RTImage();
        
        ImageMsg imageMessage = img.ToImageMsg(new HeaderMsg()); 

        // publish the message
        ros.Publish(topicName, imageMessage);
    }
    
    // Take a "screenshot" of a camera's Render Texture.
    private Texture2D RTImage()
    {
        // The Render Texture in RenderTexture.active is the one
        // that will be read by ReadPixels.
        var currentRT = RenderTexture.active;
        RenderTexture.active = imageCamera.targetTexture;

        // Render the camera's view.
        imageCamera.Render();

        // Make a new texture and read the active Render Texture into it.
        Texture2D image = new Texture2D(width, height, TextureFormat.RGB24, false);
        image.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        image.Apply();

        // Replace the original active Render Texture.
        RenderTexture.active = currentRT;
        return image;
    }
    
     //private Texture2D RTImage()
    // {
  //   	    RenderTexture targetTexture = imageCamera.targetTexture;
//	    if (targetTexture == null)
//	    {
//		targetTexture = new RenderTexture(width, height, 24);
//		imageCamera.targetTexture = targetTexture;
//	    }
//	    Texture2D texture = new Texture2D(targetTexture.width, targetTexture.height, TextureFormat.RGB24, false);
//	    imageCamera.Render();
//	    RenderTexture.active = targetTexture;
//	    texture.ReadPixels(new Rect(0, 0, targetTexture.width, targetTexture.height), 0, 0);
//	    texture.Apply();
//	    RenderTexture.active = null;
//	    return texture;
 //
     //}
}
