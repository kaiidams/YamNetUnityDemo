using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

namespace YamNetUnity
{
    public class YamNet : MonoBehaviour
    {
        private const int NumClasses = 521;
        private const int AudioBufferLengthSec = 10;

        public NNModel modelAsset;
        public UnityEvent<int, string, float> OnResult { get; private set; }

        public void StartMicrophone()
        {
            int minFreq;
            int maxFreq;

            foreach (var device in Microphone.devices)
            {
                Microphone.GetDeviceCaps(device, out minFreq, out maxFreq);
                Debug.Log($"Name: {device} MinFreq: {minFreq} MaxFreq: {maxFreq}");
            }

            string microphoneDeviceName = Microphone.devices[0];
            Microphone.GetDeviceCaps(microphoneDeviceName, out minFreq, out maxFreq);
            this.sampleRate = 48000; // AudioFeatureBuffer.SamplingRate;
            if (minFreq != 0 && maxFreq != 0)
            {
                this.sampleRate = Mathf.Clamp(this.sampleRate, minFreq, maxFreq);
            }

            this.clip = Microphone.Start(microphoneDeviceName, true, AudioBufferLengthSec, this.sampleRate);
            this.featureBuffer = new AudioFeatureBuffer();
            this.audioOffset = 0;
        }

        public void SendInput(float[] waveform)
        {
            waveform = featureBuffer.Resample(waveform, sampleRate);
            int offset = 0;
            while (offset < waveform.Length)
            {
                int written = this.featureBuffer.Write(waveform, offset, waveform.Length - offset);
                offset += written;
                while (this.featureBuffer.OutputCount >= 96 * 64)
                {
                    try
                    {
                        var features = new float[96 * 64];
                        Array.Copy(this.featureBuffer.OutputBuffer, 0, features, 0, 96 * 64);
                        this.OnPatchReceived(features);
                    }
                    finally
                    {
                        this.featureBuffer.ConsumeOutput(48 * 64);
                    }
                }
            }        
        }

        private Model model;
        private IWorker worker;
        private AudioClip clip;
        private string microphoneDeviceName;
        private int audioOffset;
        private AudioFeatureBuffer featureBuffer;
        private string[] classMap;
        private int sampleRate;

        private void Awake()
        {
            this.OnResult = new UnityEvent<int, string, float>();
        }

        // Start is called before the first frame update
        void Start()
        {
            if (modelAsset)
            {
                model = ModelLoader.Load(modelAsset);
                worker = WorkerFactory.CreateWorker(model, WorkerFactory.Device.GPU);
            }

            this.classMap = new string[NumClasses];

            TextAsset classMapData = (TextAsset)Resources.Load("yamnet_class_map", typeof(TextAsset));
            using (var reader = new StringReader(classMapData.text))
            {
                string line = reader.ReadLine(); // Discard the first line.
                while ((line = reader.ReadLine()) != null)
                {
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        string[] parts = line.Split(',');
                        int classId = int.Parse(parts[0]);
                        this.classMap[classId] = parts[2];
                    }
                }
            }
        }

        // Update is called once per frame
        void Update()
        {
            int pos = Microphone.GetPosition(microphoneDeviceName);
            if (pos < audioOffset)
            {
                pos = clip.samples;
            }
            if (pos > audioOffset)
            {
                float[] data = new float[pos - audioOffset];
                this.clip.GetData(data, this.audioOffset);
                this.audioOffset = pos;
                if (this.audioOffset >= clip.samples)
                {
                    this.audioOffset = 0;
                }
                this.SendInput(data);
            }
        }

        private void OnPatchReceived(float[] features)
        {
            print("Consume");

            if (worker != null)
            {
                Tensor inputTensor = null;

                var shape = new int[4] { 1, 96, 64, 1 };
                var inputs = new Dictionary<string, Tensor>();

                string inputName = model.inputs[0].name;
                inputTensor = new Tensor(shape, features);
                print(inputTensor[0, 34, 23, 0]);
                inputs.Add(inputName, inputTensor);
                worker.Execute(inputs);

                try
                {
                    string outputName = model.outputs[0];
                    //print(outputName);
                    Tensor output = worker.PeekOutput(outputName);
                    //print(output.name);
                    float[] predictions = output.AsFloats();
                    int bestClassId = -1;
                    float bestScore = -1000;
                    for (int i = 0; i < predictions.Length; i++)
                    {
                        if (bestScore < output[0, 0, 0, i])
                        {
                            bestScore = output[0, 0, 0, i];
                            bestClassId = i;
                        }
                    }
                    string bestClassName = this.classMap[bestClassId];
                    this.OnResult.Invoke(bestClassId, bestClassName, bestScore);
                }
                finally
                {
                    inputTensor?.Dispose();
                }
            }
        }

        public void OnDestroy()
        {
            worker?.Dispose();
        }
    }
}
