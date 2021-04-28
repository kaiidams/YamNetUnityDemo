using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using YamNetUnity;

[DisallowMultipleComponent]
[RequireComponent(typeof(YamNet))]
public class YamNetDemo : MonoBehaviour
{
    public Text text;

    // Start is called before the first frame update
    void Start()
    {
        var yamnet = GetComponent<YamNet>();
        yamnet.OnResult.AddListener(YamNetResultCallback);
        yamnet.StartMicrophone();
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void YamNetResultCallback(int bestClassId, string bestClassName, float bestScore)
    {
        float time = Time.time;
        string status = $"time: {time}, bestClassId: {bestClassId}, score: {bestScore}, bestClassName: {bestClassName}";
        Debug.Log(status);
        this.text.text = status;
    }
}
