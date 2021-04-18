using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YamNetCSharpDemo
{
    class AudioFeatureBuffer
    {
        private Mfcc _mfcc;
        private int _outputCount;
        private float[] _outputBuffer;
        private int _stftHopLength = 160;
        private int _stftWindowLength = 400;
        private int _nMelBands = 64;
        private float[] _waveformBuffer;
        private int _waveformCount;
        private int _patchWindowLength;
        private int _patchHopLength;

        public AudioFeatureBuffer(int stftHopLength = 160, int patchWindowLength = 15360, int patchHopLength = 48)
        {
            _mfcc = new Mfcc();
            _outputCount = 0;
            _waveformBuffer = new float[2 * _stftHopLength + _stftWindowLength];
            _outputBuffer = new float[_nMelBands * (_stftWindowLength + _stftHopLength)];
            _stftHopLength = stftHopLength;
            _patchWindowLength = patchWindowLength;
            _patchHopLength = patchHopLength;
        }

        public int OutputCount { get { return _outputCount; } }
        public float[] OutputBuffer { get { return _outputBuffer; } }

        public int Write(float[] waveform, int offset, int count)
        {
            int written = 0;
            if (OutputCount >= _patchWindowLength)
            {
                // Consume output buffer before writing more waveform.
                return written;
            }

            if (_waveformCount > 0)
            {
                int needed = ((_waveformCount + _stftHopLength - 1) / _stftHopLength) * _stftHopLength + _stftWindowLength - _waveformCount;
                written = Math.Min(needed, count);

                Array.Copy(waveform, offset, _waveformBuffer, _waveformCount, written);
                _waveformCount += written;

                int wavebufferOffset = 0;
                while (wavebufferOffset + _stftWindowLength < _waveformCount)
                {
                    _mfcc.Transform(_waveformBuffer, wavebufferOffset, _outputBuffer, _outputCount);
                    _outputCount += _nMelBands;
                    wavebufferOffset += _stftHopLength;
                }

                if (written < needed)
                {
                    Array.Copy(_waveformBuffer, wavebufferOffset, _waveformBuffer, 0, _waveformCount - wavebufferOffset);
                    _waveformCount -= wavebufferOffset;
                    return written;
                }

                _waveformCount = 0;
                written -= _stftWindowLength - _stftHopLength;
            }

            while (written + _stftWindowLength < count)
            {
                if (_outputCount + _nMelBands >= _outputBuffer.Length)
                {
                    return written;
                }
                _mfcc.Transform(waveform, offset + written, _outputBuffer, _outputCount);
                _outputCount += _nMelBands;
                written += _stftHopLength;
            }

            Array.Copy(waveform, offset + written, _waveformBuffer, 0, count - written);
            _waveformCount = count - written;
            written = count;
            return written;
        }

        public void Consume(int count)
        {
            Array.Copy(_outputBuffer, count, _outputBuffer, 0, _outputCount - count);
            _outputCount -= count;
        }
    }
}
