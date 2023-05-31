#include <opencv2/opencv.hpp>
#include <iostream>

AKAZE akaze = cv2.AKAZE.Create();
KeyPoint[] keyPointsLeft;
KeyPoint[] keyPointsRight;

Mat descriptorLeft = new Mat();
Mat descriptorRight = new Mat();

DescriptorMatcher matcher; // マッチング方法
DMatch[] matches;          // 特徴量ベクトル同士のマッチング結果を格納する配列

// 画像をグレースケールとして読み込む
Mat Lsrc = new Mat(sLeftPictureFile, ImreadModes.Color);

// 画像をグレースケールとして読み込む
Mat Rsrc = new Mat(sRightPictureFile, ImreadModes.Color);

// 特徴量の検出と特徴量ベクトルの計算
akaze.DetectAndCompute(Lsrc, null, out keyPointsLeft, descriptorLeft);
akaze.DetectAndCompute(Rsrc, null, out keyPointsRight, descriptorRight);

// 画像1の特徴点をoutput1に出力
Cv2.DrawKeypoints(Lsrc, keyPointsLeft, tokuLeft);
Image imageLeftToku = BitmapConverter.ToBitmap(tokuLeft);
pictureBox3.SizeMode = PictureBoxSizeMode.Zoom;
pictureBox3.Image = imageLeftToku;

// 画像2の特徴点をoutput1に出力
Cv2.DrawKeypoints(Rsrc, keyPointsRight, tokuRight);
Image imageRightToku = BitmapConverter.ToBitmap(tokuRight);
pictureBox4.SizeMode = PictureBoxSizeMode.Zoom;
pictureBox4.Image = imageRightToku;

// 総当たりでマッチング
matcher = DescriptorMatcher.Create("BruteForce");
matches = matcher.Match(descriptorLeft, descriptorRight);
Cv2.DrawMatches(Lsrc, keyPointsLeft, Rsrc, keyPointsRight, matches, output);

int size = matches.Count();
var getPtsSrc = new Vec2f[size];
var getPtsTarget = new Vec2f[size];

int count = 0;
foreach (var item in matches)
{
    var ptSrc = keyPointsLeft[item.QueryIdx].Pt;
    var ptTarget = keyPointsRight[item.TrainIdx].Pt;
    getPtsSrc[count][0] = ptSrc.X;
    getPtsSrc[count][1] = ptSrc.Y;
    getPtsTarget[count][0] = ptTarget.X;
    getPtsTarget[count][1] = ptTarget.Y;
    count++;
}

// SrcをTargetにあわせこむ変換行列homを取得する。ロバスト推定法はRANZAC。
var hom = Cv2.FindHomography(
    InputArray.Create(getPtsSrc),
    InputArray.Create(getPtsTarget),
    HomographyMethods.Ransac);

// 行列homを用いてSrcに射影変換を適用する。
Mat WarpedSrcMat = new Mat();
Cv2.WarpPerspective(
    Lsrc, WarpedSrcMat, hom,
    new OpenCvSharp.Size(Rsrc.Width, Rsrc.Height));

// 左右両方の画像を各チャンネルごとに分割
Mat LmatFloat = new Mat();
WarpedSrcMat.ConvertTo(LmatFloat, MatType.CV_16SC3);
Mat[] LmatPlanes = LmatFloat.Split();

Mat RmatFloat = new Mat();
Rsrc.ConvertTo(RmatFloat, MatType.CV_16SC3);
Mat[] RmatPlanes = RmatFloat.Split();

Mat diff0 = new Mat();
Mat diff1 = new Mat();
Mat diff2 = new Mat();

// 分割したチャンネルごとに差分を出す
Cv2.Absdiff(LmatPlanes[0], RmatPlanes[0], diff0);
Cv2.Absdiff(LmatPlanes[1], RmatPlanes[1], diff1);
Cv2.Absdiff(LmatPlanes[2], RmatPlanes[2], diff2);

// ブラーでノイズ除去
Cv2.MedianBlur(diff0, diff0, 5);
Cv2.MedianBlur(diff1, diff1, 5);
Cv2.MedianBlur(diff2, diff2, 5);

Mat wiseMat = new Mat();
Cv2.BitwiseOr(diff0, diff1, wiseMat);
Cv2.BitwiseOr(wiseMat, diff2, wiseMat);
