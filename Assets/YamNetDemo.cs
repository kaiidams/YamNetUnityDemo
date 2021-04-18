using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Barracuda;
using UnityEngine;

public class YamNetDemo : MonoBehaviour
{
    public NNModel modelAsset2;
    private Model model2;
    private IWorker worker2;
    private AudioClip clip;

    // Start is called before the first frame update
    void Start()
    {
        foreach (var device in Microphone.devices)
        {
            Debug.Log("Name: " + device);
        }

        this.clip = Microphone.Start("Built-in Microphone", true, 10, 16000);

        if (modelAsset2)
        {
            model2 = ModelLoader.Load(modelAsset2);
            worker2 = WorkerFactory.CreateWorker(model2, WorkerFactory.Device.CPU);
        }
    }

    // Update is called once per frame
    void Update()
    {
        print($"{clip.length} {clip.samples}");
        return;

        if (worker2 != null)
        {
            Tensor inputTensor = null;

            var x = new float[96 * 64];
            for (int i = 0; i < x.Length; i++) x[i] = -5.0f;
            var s = new int[4] { 1, 96, 64, 1 };
            var inputs = new Dictionary<string, Tensor>();

            string name2 = model2.inputs[0].name;
            inputTensor = new Tensor(s, x);
            print(inputTensor[0, 34, 23, 0]);
            inputs.Add(name2, inputTensor);
            worker2.Execute(inputs);

            try
            {
                string name = model2.outputs[0];
                print(name);
                Tensor output = worker2.PeekOutput(name);
                print(output.name);
                float[] features = output.AsFloats();
                print(string.Format("Shape {0} {1} {2} {3} {4}",
                    output.dimensions,
                    output.shape[0], output.shape[1], output.shape[2], output.shape[3]));
                print(string.Format("Shape {0} {1} {2} {3}",
                    output.batch, output.height, output.width, output.channels));
                print(features.Length);
                int mi = -1;
                float mv = -1000;
                for (int i = 0; i < features.Length; i++)
                {
                    if (mv < output[0, 0, 0, i])
                    {
                        mv = output[0, 0, 0, i];
                        mi = i;
                    }
                }
                Debug.Log($"mi: {mi}, mv: {mv}");
            }
            finally
            {
                inputTensor?.Dispose();
                inputTensor = null;
            }
        }
    }

    public void OnDestroy()
    {
        worker2?.Dispose();
    }
}
