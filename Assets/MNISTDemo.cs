using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;
using Random = UnityEngine.Random;

public class MNISTDemo : MonoBehaviour
{
    public NNModel modelAsset;
    private Model model;
    private IWorker worker;

    // Start is called before the first frame update
    void Start()
    {
        if (modelAsset)
        {
            model = ModelLoader.Load(modelAsset);
            worker = WorkerFactory.CreateWorker(model, WorkerFactory.Device.GPU);
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (worker != null)
        {
            int index = (int)(Random.value * 4.0);
            var data = MakeData(index);
            Tensor inputTensor = data.Item1;
            int label = data.Item2;

            var inputs = new Dictionary<string, Tensor>();

            try
            {
                string inputName = model.inputs[0].name;
                inputs.Add(inputName, inputTensor);
                worker.Execute(inputs);

                string outputName = model.outputs[0];
                print($"input: {inputName}, output: {outputName}");
                Tensor output = worker.PeekOutput(outputName);

                float[] features = output.AsFloats();
                var result = ArgMax(features);
                Debug.Log($"target: {label} predict: {result.Item1}, score: {result.Item2}");
            }
            finally
            {
                inputTensor.Dispose();
            }
        }
    }

    Tuple<Tensor, int> MakeData(int index)
    {
        int[] shape = { 1, 28, 28, 1 };
        float[] x = new float[28 * 28];
        for (int i = 0; i < 28 * 28; i++)
        {
            x[i] = MNISTData.X[index, i] / 255.0f;
        }
        int y = MNISTData.Y[index];
        return new Tuple<Tensor, int>(new Tensor(shape, x), y);
    }

    Tuple<int, float> ArgMax(float[] scores)
    {
        int mi = -1;
        float mv = -1000;
        for (int i = 0; i < scores.Length; i++)
        {
            if (mv < scores[i])
            {
                mv = scores[i];
                mi = i;
            }
        }
        return new Tuple<int, float>(mi, mv);
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }
}
