﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

public class MNISTData
{
    public static byte[,] X = {
        {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 43, 39, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 227, 252, 247, 142, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 233, 232, 135, 197, 241, 253, 252, 252, 247, 188, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 221, 253, 252, 252, 252, 226, 84, 84, 212, 252, 182, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 54, 186, 252, 253, 252, 155, 121, 24, 0, 0, 215, 252, 147, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 253, 253, 194, 27, 0, 0, 0, 0, 0, 233, 253, 129, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 168, 124, 0, 0, 0, 0, 0, 0, 100, 247, 252, 42, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 190, 252, 185, 4, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 232, 252, 84, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 236, 252, 190, 14, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 43, 123, 201, 253, 253, 255, 253, 200, 148, 236, 255, 239, 62, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 84, 190, 253, 252, 252, 252, 252, 253, 252, 252, 252, 252, 253, 89, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 82, 246, 252, 232, 231, 134, 126, 38, 21, 151, 252, 252, 252, 253, 63, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 84, 246, 252, 155, 0, 0, 0, 0, 0, 126, 232, 252, 252, 252, 253, 154, 6, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 190, 252, 155, 7, 0, 0, 0, 27, 150, 253, 252, 199, 77, 174, 253, 252, 109, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 61, 253, 214, 0, 0, 0, 32, 96, 218, 253, 247, 131, 53, 0, 0, 237, 253, 191, 14, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 183, 252, 126, 0, 36, 103, 237, 252, 252, 182, 53, 0, 0, 0, 0, 55, 231, 252, 163, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 253, 252, 189, 197, 241, 253, 245, 222, 56, 4, 0, 0, 0, 0, 0, 0, 174, 251, 221, 74, 0, 0, 0, 0, 0,
            0, 0, 0, 156, 252, 252, 210, 145, 84, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 161, 252, 121, 0, 0, 0, 0, 0,
            0, 0, 0, 7, 42, 42, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 42, 7, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        },
        {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 149, 254, 175, 105, 105, 105, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 48, 230, 253, 253, 253, 253, 253, 253, 236, 174, 60, 32, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 48, 222, 253, 253, 253, 253, 253, 253, 253, 254, 253, 253, 211, 121, 5, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 253, 253, 253, 253, 253, 253, 254, 253, 253, 253, 253, 144, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 253, 253, 213, 37, 29, 118, 179, 196, 253, 253, 253, 250, 145, 4, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 75, 240, 253, 253, 253, 232, 35, 0, 0, 0, 0, 18, 83, 237, 253, 253, 253, 32, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 105, 253, 253, 253, 253, 143, 0, 0, 0, 0, 0, 0, 0, 120, 253, 253, 253, 211, 32, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 247, 253, 253, 253, 253, 215, 0, 0, 0, 0, 0, 0, 0, 7, 85, 253, 253, 253, 200, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 254, 253, 253, 253, 253, 246, 92, 0, 0, 0, 0, 0, 0, 0, 18, 196, 253, 253, 208, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 254, 253, 253, 253, 253, 253, 119, 0, 0, 0, 0, 0, 0, 0, 0, 179, 253, 253, 208, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 194, 254, 254, 254, 254, 254, 120, 0, 0, 0, 0, 0, 0, 0, 0, 31, 254, 255, 254, 105, 0, 0,
            0, 0, 0, 0, 0, 0, 105, 253, 253, 253, 200, 175, 21, 0, 0, 0, 0, 0, 0, 0, 0, 66, 253, 253, 253, 104, 0, 0,
            0, 0, 0, 0, 0, 0, 105, 253, 253, 253, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 179, 253, 253, 224, 37, 0, 0,
            0, 0, 0, 0, 0, 0, 105, 253, 253, 253, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 214, 253, 253, 208, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 12, 214, 253, 253, 161, 0, 0, 0, 0, 0, 0, 0, 0, 36, 216, 253, 253, 253, 164, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 209, 253, 253, 231, 162, 15, 0, 0, 0, 0, 8, 39, 214, 253, 253, 253, 225, 52, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 130, 253, 253, 253, 253, 190, 134, 56, 134, 135, 163, 253, 253, 253, 253, 253, 14, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 4, 117, 253, 253, 253, 253, 253, 245, 253, 255, 253, 253, 253, 253, 171, 14, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 4, 129, 248, 253, 253, 253, 253, 253, 254, 253, 253, 232, 164, 17, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 122, 253, 253, 253, 253, 255, 217, 104, 55, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        },
        {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 45, 233, 131, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 176, 19, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 141, 254, 208, 33, 0, 0, 0, 0, 0, 0, 0, 0, 212, 254, 170, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 16, 169, 254, 224, 39, 0, 0, 0, 0, 0, 0, 0, 33, 222, 254, 237, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 62, 254, 254, 131, 0, 0, 0, 0, 0, 0, 0, 0, 132, 254, 254, 237, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 215, 254, 254, 131, 0, 0, 0, 0, 0, 0, 0, 47, 238, 254, 254, 206, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 239, 254, 254, 131, 0, 0, 0, 0, 0, 0, 0, 54, 254, 254, 254, 61, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 128, 251, 254, 219, 27, 0, 0, 0, 0, 0, 0, 0, 54, 254, 254, 254, 61, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 159, 254, 254, 229, 55, 0, 0, 0, 0, 0, 0, 0, 54, 254, 254, 254, 61, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 44, 237, 254, 254, 221, 185, 185, 185, 185, 185, 185, 185, 200, 254, 254, 254, 189, 6, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 41, 215, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 157, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 49, 116, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 243, 54, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 2, 8, 148, 111, 8, 69, 184, 184, 184, 248, 254, 254, 143, 7, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 230, 254, 234, 58, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 230, 254, 219, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 230, 254, 219, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 126, 250, 254, 219, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 254, 254, 139, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 254, 254, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 254, 254, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 200, 78, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        },
        {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 113, 161, 161, 52, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 121, 235, 236, 254, 254, 187, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47, 175, 254, 159, 32, 3, 165, 231, 0, 0, 24, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 216, 251, 99, 2, 0, 0, 27, 210, 51, 178, 132, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 171, 255, 78, 0, 0, 0, 0, 16, 236, 254, 124, 10, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 253, 152, 1, 0, 0, 0, 30, 177, 253, 151, 8, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 254, 89, 0, 4, 109, 175, 247, 211, 81, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 104, 254, 69, 104, 211, 254, 237, 128, 19, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 123, 254, 249, 253, 204, 95, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 242, 254, 243, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 99, 253, 217, 196, 246, 77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 23, 161, 235, 96, 6, 36, 195, 225, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 4, 241, 242, 52, 0, 0, 0, 86, 254, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 5, 199, 234, 49, 0, 0, 0, 0, 106, 254, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 20, 151, 254, 56, 0, 0, 0, 0, 7, 198, 248, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 161, 236, 64, 4, 4, 53, 82, 223, 252, 126, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 48, 205, 254, 254, 254, 254, 252, 189, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 24, 91, 91, 91, 61, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        },
    };

    public static byte[] Y = { 2, 0, 4, 8 };
}
