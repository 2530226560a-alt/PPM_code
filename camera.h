#pragma once
#include "cameraModel.h"
#include <opencv2/opencv.hpp> 
#include <string>
#include <stdio.h>
#include<vector>
using namespace std;
using namespace cv;

class Camera
{
public:
	Etiseo::CameraModel *cam;
	cv::Mat fore;
	cv::Mat map;
	cv::Mat maph;
	cv::Mat top;
	cv::Mat topView;
	cv::Mat topshow;
	cv::Mat frame;
	cv::Mat frm;
	cv::Mat foreground;
	cv::Mat foreground2;
	cv::Mat rctMap;
	VideoCapture capture;
	int FrameNumber;
	cv::Mat M;
	cv::Mat M3;
	cv::Mat Mnew;
	bool flag;
	Rect r[400];
	vector<vector<Point>> contours;
	vector<vector<Point>> contoursb;
	Mat pF;
	Mat T;
	int Tnum;
	double prob[400];
	double probB[400];
	double probT[400];
	double probM[400];

	double probI[400];

	int idx[400];
	Point p[400];
	vector <Rect> rb;
	vector <Rect> rba;
	Mat foreground3;
	cv::Mat foregroundread;
	cv::Mat foregroundread2;
	cv::Mat frameread;
	Point top1, top2, top3;
	Mat temp;
	vector<vector<int>> templine;
	Mat ele;

	Mat temphist;
	Mat temphist2;

	cv::Mat mapE;
	vector<vector < vector<int> >> line;
	Point  vp;

#define aratio 0.4
#define upr 0.9
#define PHeight 2000

	Mat roi33;
	Mat roidoublem33;
	Mat rst33;
	Mat rst111;
	double tempSum = 0;
	double tempMax = 0;
	double temF;
	double temB;
	double  tableG[2][16] = { { 0.5, 0.5793, 0.6554, 0.7257, 0.7881, 0.8413, 0.8849, 0.9192, 0.9452, 0.9641, 0.9772, 0.9861, 0.9918, 0.9953, 0.9974, 0.9987 },
{ 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, } };

	Scalar colorful[40] = { CV_RGB(255, 128, 128), CV_RGB(255, 255, 128), CV_RGB(128, 0, 64), CV_RGB(128, 0, 255), CV_RGB(128, 255, 255), CV_RGB(0, 128, 255), CV_RGB(255, 0, 128), CV_RGB(255, 128, 255),
CV_RGB(128, 255, 128), CV_RGB(128, 128, 255), CV_RGB(0, 64, 128), CV_RGB(0, 128, 128), CV_RGB(0, 255, 0), CV_RGB(255, 128, 64), CV_RGB(128, 64, 64), CV_RGB(255, 0, 0), CV_RGB(255, 255, 0),
CV_RGB(0,255,255), CV_RGB(0, 255, 0), CV_RGB(255, 255, 255),
CV_RGB(255, 128, 128), CV_RGB(255, 255, 128), CV_RGB(128, 0, 64), CV_RGB(128, 0, 255), CV_RGB(128, 255, 255), CV_RGB(0, 128, 255), CV_RGB(255, 0, 128), CV_RGB(255, 128, 255),
CV_RGB(128, 255, 128), CV_RGB(128, 128, 255), CV_RGB(0, 64, 128), CV_RGB(0, 128, 128), CV_RGB(0, 255, 0), CV_RGB(255, 128, 64), CV_RGB(128, 64, 64), CV_RGB(255, 0, 0), CV_RGB(255, 255, 0),
CV_RGB(64, 0, 64), CV_RGB(0, 255, 128), CV_RGB(255, 255, 255) };

	Camera(string videoLocation, string bgLocation = "")
	{
		FrameNumber = 0;
		cam = new Etiseo::CameraModel;

		capture.open(videoLocation);
		if (!capture.isOpened())
			cout << "Fail to open!  " << videoLocation << endl;

		capture.read(frame);

		top = Mat::zeros(1000, 1000, CV_8UC1);
		topshow = Mat::zeros(1000, 1000, CV_8UC1);
		topView = Mat::zeros(1000, 1000, CV_8UC1);
		map = Mat::zeros(1000, 1000, CV_32FC2)*(-1);
		maph = Mat::ones(1000, 1000, CV_32FC2)*(-1);
		mapE = Mat::zeros(1000, 1000, CV_32FC2)*(-1);
		temp = Mat::zeros(1000, 1000, CV_8UC1);
		rctMap = Mat::zeros(1000, 1000, CV_32FC4);
		M = Mat(3, 3, CV_32F);
		M3 = Mat(3, 1, CV_32F);
		ele = getStructuringElement(MORPH_RECT, Size(3, 3));

		pF = Mat::zeros(frame.rows, frame.cols, CV_32S);
		T = Mat::zeros(10000, 3, CV_32S);

		temphist = Mat::zeros(1, frame.cols, CV_32FC1);
		temphist2 = Mat::zeros(1, frame.cols, CV_32FC1);
		foregroundread = Mat::zeros(288, 360, CV_8UC1);
		foregroundread2 = Mat::zeros(288, 360, CV_8UC1);
		foreground = Mat::zeros(288, 360, CV_8UC1);
		foreground2 = Mat::zeros(288, 360, CV_8UC1);

	}


	void readNextFrame()
	{
		capture.read(frame);
		FrameNumber++;
	}

	void setFrameNumber(int n)
	{
		capture.set(CV_CAP_PROP_POS_FRAMES, n);
		FrameNumber = n;
	}

	void readFrameNumber(int n)
	{
		capture.set(CV_CAP_PROP_POS_FRAMES, n);
		capture.read(frame);
		capture.set(CV_CAP_PROP_POS_FRAMES, FrameNumber);
	}

	//load the processed mask images
	void maskfore0()
	{
		string base = ".\\PPM_code\\Terrace_binary Masks_4 cameras\\C0\\";
		string filename = base + to_string(FrameNumber) + ".bmp";
		foreground.release();
		foregroundread = imread(filename, cv::IMREAD_COLOR);
		if (foregroundread.empty()) {
			foreground = Mat::zeros(288, 360, CV_8UC1);
		}
		else {
			cvtColor(foregroundread, foreground, COLOR_BGR2GRAY);
			threshold(foreground, foreground, 100, 255, cv::THRESH_BINARY);
		}

		string filenames = base + to_string(FrameNumber) + ".bmp";
		foreground2.release();
		foregroundread2 = imread(filenames, cv::IMREAD_COLOR);
		if (foregroundread2.empty()) {
			foreground2 = Mat::zeros(288, 360, CV_8UC1);
		}
		else {
			cvtColor(foregroundread2, foreground2, COLOR_BGR2GRAY);
		}
	}
	void maskfore1()
	{
		string base = ".\\PPM_code\\Terrace_binary Masks_4 cameras\\C1\\";
		string filename = base + to_string(FrameNumber) + ".bmp";
		foreground.release();
		foregroundread = imread(filename, cv::IMREAD_COLOR);
		if (foregroundread.empty()) {
			foreground = Mat::zeros(288, 360, CV_8UC1);
		}
		else {
			cvtColor(foregroundread, foreground, COLOR_BGR2GRAY);
			threshold(foreground, foreground, 100, 255, cv::THRESH_BINARY);
		}

		string filenames = base + to_string(FrameNumber) + ".bmp";
		foreground2.release();
		foregroundread2 = imread(filenames, cv::IMREAD_COLOR);
		if (foregroundread2.empty()) {
			foreground2 = Mat::zeros(288, 360, CV_8UC1);
		}
		else {
			cvtColor(foregroundread2, foreground2, COLOR_BGR2GRAY);
		}
	}
	void maskfore2()
	{
		string base = ".\\PPM_code\\Terrace_binary Masks_4 cameras\\C2\\";
		string filename = base + to_string(FrameNumber) + ".bmp";
		foreground.release();
		foregroundread = imread(filename, cv::IMREAD_COLOR);
		if (foregroundread.empty()) {
			foreground = Mat::zeros(288, 360, CV_8UC1);
		}
		else {
			cvtColor(foregroundread, foreground, COLOR_BGR2GRAY);
			threshold(foreground, foreground, 100, 255, cv::THRESH_BINARY);
		}

		string filenames = base + to_string(FrameNumber) + ".bmp";
		foreground2.release();
		foregroundread2 = imread(filenames, cv::IMREAD_COLOR);
		if (foregroundread2.empty()) {
			foreground2 = Mat::zeros(288, 360, CV_8UC1);
		}
		else {
			cvtColor(foregroundread2, foreground2, COLOR_BGR2GRAY);
		}
	}
	void maskfore3()
	{
		string base = ".\\PPM_code\\Terrace_binary Masks_4 cameras\\C3\\";
		string filename = base + to_string(FrameNumber) + ".bmp";
		foreground.release();
		foregroundread = imread(filename, cv::IMREAD_COLOR);
		if (foregroundread.empty()) {
			foreground = Mat::zeros(288, 360, CV_8UC1);
		}
		else {
			cvtColor(foregroundread, foreground, COLOR_BGR2GRAY);
			threshold(foreground, foreground, 100, 255, cv::THRESH_BINARY);
		}

		string filenames = base + to_string(FrameNumber) + ".bmp";
		foreground2.release();
		foregroundread2 = imread(filenames, cv::IMREAD_COLOR);
		if (foregroundread2.empty()) {
			foreground2 = Mat::zeros(288, 360, CV_8UC1);
		}
		else {
			cvtColor(foregroundread2, foreground2, COLOR_BGR2GRAY);
		}
	}

	void mapToTop()
	{
		double X, Y, Z = 0;
		double x, y;
		double xt, yt;
		double a, b, c, d, e, f;
		double aa, bb, cc, dd;
		bool flag1, flag2, flag3, flag4;
		float overlap;

		for (int i = 0; i < 1000; i = i + 1)
		{
			for (int j = 0; j < 1000; j = j + 1)
			{
				X = (i - 250) * 30;
				Y = (j - 250) * 30;
				Z = 0;
				cam->worldToImage(X, Y, Z, x, y);
				x = x / 2;
				y = y / 2;
				Z = PHeight;
				cam->worldToImage(X, Y, Z, xt, yt);
				xt = xt / 2;
				yt = yt / 2;
				a = x - abs(y - yt)*0.35*0.5;
				b = yt;
				c = abs(y - yt)*0.35;
				d = y - yt;
				e = a + c;
				f = b + d;
				Z = 1200;
				cam->worldToImage(X, Y, Z, xt, yt);
				xt = xt / 2;
				yt = yt / 2;
				if (x > 0 && y > 0 && x < foreground.cols&&y < foreground.rows)
				{
					topshow.at<uchar>(i, j) = 1;
				}

				if (d > 0)
				{
					flag1 = (a > 0 && a < foreground.cols) && (b > 0 && b < foreground.rows);
					flag2 = (b > 0 && b < foreground.rows) && (e > 0 && e < foreground.cols);
					flag3 = (a > 0 && a < foreground.cols) && (f > 0 && f < foreground.rows);
					flag4 = (f > 0 && f < foreground.rows) && (e > 0 && e < foreground.cols);
					if (flag1 || flag2 || flag3 || flag4)
					{
						top.at<uchar>(i, j) = 1;

						map.at<Point2f>(i, j).x = x;
						map.at<Point2f>(i, j).y = y;

						if (xt < 0)
							xt = 0;
						else if (xt > foreground.cols)
							xt = foreground.cols;

						if (yt < 0)
							yt = 0;
						else if (yt > foreground.rows)
							yt = foreground.rows;

						maph.at<Point2f>(i, j).x = xt;
						maph.at<Point2f>(i, j).y = yt;
					}
				}
				mapE.at<Point2f>(i, j).x = x;
				mapE.at<Point2f>(i, j).y = y;

				aa = a > 0 ? a : 0;
				bb = b > 0 ? b : 0;
				if (a < 0)
					c = c + a;
				cc = a + c < foreground.cols ? c : (foreground.cols - a);
				dd = b + d < foreground.rows ? d : (foreground.rows - b);
				rctMap.at<Vec4f>(i, j)[0] = aa;
				rctMap.at<Vec4f>(i, j)[1] = bb;
				rctMap.at<Vec4f>(i, j)[2] = cc;
				rctMap.at<Vec4f>(i, j)[3] = dd;
			}
		}
	}

	double dutyCycle(int i, int j)
	{
		if ((topshow.at<uchar>(i, j) > 0))
		{
			double PV = rctMap.at<Vec4f>(i, j)[2] * rctMap.at<Vec4f>(i, j)[3];
			double PS = 0;

			for (int k = rctMap.at<Vec4f>(i, j)[0]; k < (rctMap.at<Vec4f>(i, j)[0] + rctMap.at<Vec4f>(i, j)[2]); k++)
				for (int l = rctMap.at<Vec4f>(i, j)[1]; l < (rctMap.at<Vec4f>(i, j)[1] + rctMap.at<Vec4f>(i, j)[3]); l++)
				{
					PS += foreground.at<uchar>(l, k);
				}

			PS = PS / 255;
			return PS / PV;
		}
		else
			return 1;
	}

	float colOverlap(Rect box1, Rect box2)
	{
		if (box1.x > box2.x + box2.width) { return 0.0; }
		if (box1.y > box2.y + box2.height) { return 0.0; }
		if (box1.x + box1.width < box2.x) { return 0.0; }
		if (box1.y + box1.height < box2.y) { return 0.0; }
		float colInt = min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x);
		float rowInt = min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y);
		return (colInt / box1.width);
	}

	int lineti(int nFrmNum, Mat bboxData, int k, Mat topCross, Mat topCross2, Mat topCross3)
	{
		cout << "[Debug] lineti called for frame " << nFrmNum << ", camera " << k << endl;
		double X, Y, x, y, xt, yt, offset, newmid;
		double a, b, c, d, e, f;
		double tempv;
		int numbox = 0;
		int numI = 0;
		int column[20];
		double tempv1 = 0;
		int kk = 0;
		templine.clear();
		double incl;
		Point etop;
		double sX, sY, SX, SY, sigk;
		int flagn;

		double camx, camy;
		cam->worldToImage(cam->mCposx, cam->mCposy, 0, camx, camy);
		camx = camx / 2;
		camy = camy / 2;

		// collect bbox rows for current frame/camera from bboxData; if none, we'll fall back to mask-based contours
		for (int bbp = 0; bbp < 89562; bbp++)
		{
			if (bboxData.at<float>(bbp, 0) == nFrmNum)
				if (bboxData.at<float>(bbp, 1) == k)
				{
					column[numI] = bbp;
					numI += 1;

				}
		}
		cout << "[Debug] Found " << numI << " detections for frame " << nFrmNum << ", camera " << k << endl;

		// Fallback: if no detections available for this frame/camera, synthesize lines from foreground mask contours
		if (numI == 0)
		{
			cout << "[Debug] No detections found for frame " << nFrmNum << ", camera " << k << ". Using mask fallback." << endl;
			vector<vector<Point>> localContours;
			findContours(foreground, localContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			cout << "[Debug] Found " << localContours.size() << " contours in mask" << endl;
			for (auto &cnt : localContours)
			{
				Rect br = boundingRect(cnt);
				int x1 = br.x;
				int y1 = br.y;
				int width = br.width;
				int height = br.height;
				if (width < 8 || height < 14 || width * height < 100) continue;

				// compute top points using simplified vertical line anchored at bbox mid x
				cam->imageToWorld((x1 + width / 2) * 2, (y1 + height) * 2, 0, X, Y);
				top3.x = X / 30 + 250;
				top3.y = Y / 30 + 250;

				cam->imageToWorld((x1 + 2 + width / 2) * 2, (y1 + height) * 2, 0, sX, sY);
				SX = sX / 30 + 250;
				SY = sY / 30 + 250;

				sigk = abs(sqrt((top3.x - SX) * (top3.x - SX) + (top3.y - SY) * (top3.y - SY)));

				if (top3.x < 0 || top3.x > 1000 || top3.y < 0 || top3.y > 1000) continue;

				X = (top3.x - 250) * 30;
				Y = (top3.y - 250) * 30;
				cam->worldToImage(X, Y, 0, x, y);
				x = x / 2;
				y = y / 2;

				cam->worldToImage(X, Y, PHeight, xt, yt);
				xt = xt / 2;
				yt = yt / 2;

				a = x - abs(y - yt) * 0.35 * 0.5;
				b = yt;
				c = abs(y - yt) * 0.35;
				d = y - yt;
				e = a + c;
				f = b + d;

				// simplified top1/top2 along vertical line at image x, no kt/bt needed
				cam->imageToWorld(x * 2, (b + d * upr) * 2, 0, X, Y);
				top1.x = X / 30 + 250;
				top1.y = Y / 30 + 250;

				cam->imageToWorld(x * 2, (b + d * 1.1) * 2, 0, X, Y);
				top2.x = X / 30 + 250;
				top2.y = Y / 30 + 250;

				// record templine
				templine.push_back(vector<int>(15, 0));
				int kk2 = (int)templine.size() - 1;
				templine[kk2][0] = (int)(top1.x);
				templine[kk2][1] = (int)(top1.y);
				templine[kk2][2] = (int)(top2.x);
				templine[kk2][3] = (int)(top2.y);
				templine[kk2][4] = (int)(top3.x);
				templine[kk2][5] = (int)(top3.y);
				templine[kk2][6] = (int)(width);
				templine[kk2][7] = -1;
				templine[kk2][9] = (int)(k);
				templine[kk2][10] = 0;
				templine[kk2][11] = 0;
				templine[kk2][12] = 0;
				templine[kk2][13] = 0;
				templine[kk2][14] = (sigk >= 1) ? sigk * sigk : 1;

				int flagb = 0;
				if ((x1 < 10 || x1 + width > 350) && height * width < c * d * aratio) {
					flagb = 1;
				}
				if (x - 0.1 * c < 0 || x + 0.1 * c > 360 || flagb == 1) {
					templine[kk2][11] = -1;
				}
			}

			line.push_back(templine);
			return (int)templine.size();
		}

		if (numI > 0)
		{
			if (numI == 1)
			{
				int bbrow = column[0];
				int x1 = bboxData.at<float>(bbrow, 2);
				int y1 = bboxData.at<float>(bbrow, 3);
				int width = bboxData.at<float>(bbrow, 4);
				int height = bboxData.at<float>(bbrow, 5);
				float kt = bboxData.at<float>(bbrow, 6);
				float bt = bboxData.at<float>(bbrow, 7);
				int xtp = bboxData.at<float>(bbrow, 8);
				int ytp = bboxData.at<float>(bbrow, 9);

				flagn = 0;

				cam->imageToWorld((x1 + width / 2) * 2, (y1 + height) * 2, 0, X, Y);
				top3.x = X / 30 + 250;
				top3.y = Y / 30 + 250;

				cam->imageToWorld((x1 + 2 + width / 2) * 2, (y1 + height) * 2, 0, sX, sY);
				SX = sX / 30 + 250;
				SY = sY / 30 + 250;

				sigk = abs(sqrt((top3.x - SX)*(top3.x - SX)
					+ (top3.y - SY)*(top3.y - SY)));

				if (top3.x >= 0 && top3.x <= 1000 && top3.y >= 0 && top3.y <= 1000)
				{
					X = (top3.x - 250) * 30; 
					Y = (top3.y - 250) * 30;
					cam->worldToImage(X, Y, 0, x, y);
					x = x / 2;
					y = y / 2;

					X = (top3.x - 250) * 30;
					Y = (top3.y - 250) * 30;
					cam->worldToImage(X, Y, PHeight, xt, yt);
					xt = xt / 2;
					yt = yt / 2;

					a = x - abs(y - yt)*0.35*0.5;
					b = yt;
					c = abs(y - yt)*0.35;
					d = y - yt;
					e = a + c;
					f = b + d;

					if ((vp.x - xtp) != 0)
					{
						double ratt = height / d;

						if (y1 + height > foreground.rows - 10)
						{
							if (ratt < 0.9)
							{
								double kp, bp;
								kp = (vp.y - ytp) / (vp.x - xtp);
								bp = ytp - kp * xtp;

								cam->imageToWorld(((b + d * upr - bp) / kp) * 2,
									(b + d * upr) * 2, 0, X, Y);
							}
							else
							{
								cam->imageToWorld(((b + d * upr - bt) / kt) * 2,
									(b + d * upr) * 2, 0, X, Y);
							}
						}
						else
						{
							cam->imageToWorld(((b + d * upr - bt) / kt) * 2,
								(b + d * upr) * 2, 0, X, Y);
						}
						top1.x = X / 30 + 250;
						top1.y = Y / 30 + 250;

						if (y1 + height > foreground.rows - 10)
						{
							if (ratt < 0.9)
							{
								double kp, bp;
								kp = (vp.y - ytp) / (vp.x - xtp);
								bp = ytp - kp * xtp;

								cam->imageToWorld(((camy - bp) / kp) * 2,
									camy * 2, 0, X, Y);
							}
							else
							{
								cam->imageToWorld(((camy - bt) / kt) * 2,
									camy * 2, 0, X, Y);
							}

						}
						else
						{
							cam->imageToWorld(((b + d * 1.1 - bt) / kt) * 2,
								(b + d * 1.1) * 2, 0, X, Y);
						}
						top2.x = X / 30 + 250;
						top2.y = Y / 30 + 250;

						if (y1 + height > foreground.rows - 10 && ratt < 0.9)
						{
							double kp, bp;
							kp = (vp.y - ytp) / (vp.x - xtp);
							bp = ytp - kp * xtp;

							cam->imageToWorld(((b + d - bp) / kp) * 2,
								(b + d) * 2, 0, X, Y);
							top3.x = X / 30 + 250;
							top3.y = Y / 30 + 250;
						}
						else
						{
							cam->imageToWorld(((b + d - bt) / kt) * 2,
								(b + d) * 2, 0, X, Y);
							top3.x = X / 30 + 250;
							top3.y = Y / 30 + 250;
						}

						templine.push_back(vector<int>(15, 0));
						templine[kk][0] = (int)(top1.x);
						templine[kk][1] = (int)(top1.y);
						templine[kk][2] = (int)(top2.x);
						templine[kk][3] = (int)(top2.y);
						templine[kk][4] = (int)(top3.x);
						templine[kk][5] = (int)(top3.y);
						templine[kk][6] = (int)(width);
						templine[kk][7] = -1;
						templine[kk][9] = (int)(k);
						templine[kk][10] = 0;
						templine[kk][11] = 0;
						templine[kk][12] = 0;
						templine[kk][13] = 0;
						if (sigk >= 1)
						{
							templine[kk][14] = sigk * sigk;
						}
						else
						{
							templine[kk][14] = 1;
						}

						if (y1 + height > foreground.rows - 10 && ratt < 0.9)
						{
							templine[kk][12] = -1;
						}

						int flagb = 0;
						if ((x1 < 10 || x1 + width > 350) && height * width < c * d * aratio)
						{
							flagb = 1;
						}

						if (x - 0.1*c < 0 || x + 0.1*c > 360 || flagb == 1)
						{
							templine[kk][11] = -1;
						}

						if (x + c / 2 >= 360 && templine[kk][11] != -1 && templine[kk][12] != -1)
						{
							offset = (x - c / 2) - x1;
							newmid = x - offset;

							cam->imageToWorld(newmid * 2, (y1 + height) * 2, 0, X, Y);
							top3.x = X / 30 + 250;
							top3.y = Y / 30 + 250;

							cam->imageToWorld(newmid * 2,
								(b + d * 0.9) * 2, 0, X, Y);
							top1.x = X / 30 + 250;
							top1.y = Y / 30 + 250;

							cam->imageToWorld(newmid * 2,
								(b + d * 1.1) * 2, 0, X, Y);
							top2.x = X / 30 + 250;
							top2.y = Y / 30 + 250;

							templine[kk][0] = (int)(top1.x);
							templine[kk][1] = (int)(top1.y);
							templine[kk][2] = (int)(top2.x);
							templine[kk][3] = (int)(top2.y);
							templine[kk][4] = (int)(top3.x);
							templine[kk][5] = (int)(top3.y);

							flagn = 1;
						}

						else if (x - c / 2 <= 0 && templine[kk][11] != -1 && templine[kk][12] != -1)
						{
							offset = (x + c / 2) - (x1 + width);
							newmid = x - offset;

							cam->imageToWorld(newmid * 2, (y1 + height) * 2, 0, X, Y);
							top3.x = X / 30 + 250;
							top3.y = Y / 30 + 250;

							cam->imageToWorld(newmid * 2,
								(b + d * 0.9) * 2, 0, X, Y);
							top1.x = X / 30 + 250;
							top1.y = Y / 30 + 250;

							cam->imageToWorld(newmid * 2,
								(b + d * 1.1) * 2, 0, X, Y);
							top2.x = X / 30 + 250;
							top2.y = Y / 30 + 250;

							templine[kk][0] = (int)(top1.x);
							templine[kk][1] = (int)(top1.y);
							templine[kk][2] = (int)(top2.x);
							templine[kk][3] = (int)(top2.y);
							templine[kk][4] = (int)(top3.x);
							templine[kk][5] = (int)(top3.y);

							flagn = 1;
						}

						if (y1 + height >= foreground.rows - 10 && ratt < 0.9)
						{
							if (templine[kk][11] != -1 && flagn == 0)
							{
								double kp, bp;
								kp = (vp.y - ytp) / (vp.x - xtp);
								bp = ytp - kp * xtp;
							}

						}

						kk++;
					}
					else
					{
						double ratt = height / d;

						if (y1 + height > foreground.rows - 10 && ratt < 0.9)
						{
							cam->imageToWorld(x * 2,
								(b + d * upr) * 2, 0, X, Y);
						}
						else
						{
							cam->imageToWorld(((b + d * upr - bt) / kt) * 2,
								(b + d * upr) * 2, 0, X, Y);
						}
						top1.x = X / 30 + 250;
						top1.y = Y / 30 + 250;

						if (y1 + height > foreground.rows - 10)
						{
							if (ratt < 0.9)
							{
								cam->imageToWorld(x * 2,
									camy * 2, 0, X, Y);
							}
							else
							{
								cam->imageToWorld(x * 2,
									camy * 2, 0, X, Y);
							}
						}
						else
						{
							cam->imageToWorld(((b + d * 1.1 - bt) / kt) * 2,
								(b + d * 1.1) * 2, 0, X, Y);
						}
						top2.x = X / 30 + 250;
						top2.y = Y / 30 + 250;

						if (y1 + height > foreground.rows - 10 && ratt < 0.9)
						{
							cam->imageToWorld(x * 2,
								(b + d) * 2, 0, X, Y);
							top3.x = X / 30 + 250;
							top3.y = Y / 30 + 250;
						}
						else
						{
							cam->imageToWorld(((b + d - bt) / kt) * 2,
								(b + d) * 2, 0, X, Y);
						}

						templine.push_back(vector<int>(15, 0));
						templine[kk][0] = (int)(top1.x);
						templine[kk][1] = (int)(top1.y);
						templine[kk][2] = (int)(top2.x);
						templine[kk][3] = (int)(top2.y);
						templine[kk][4] = (int)(top3.x);
						templine[kk][5] = (int)(top3.y);
						templine[kk][6] = (int)(width);
						templine[kk][7] = -1;
						templine[kk][9] = (int)(k);
						templine[kk][10] = 0;
						templine[kk][11] = 0;
						templine[kk][12] = 0;
						templine[kk][13] = 0; 
						if (sigk >= 1)
						{
							templine[kk][14] = sigk * sigk;
						}
						else
						{
							templine[kk][14] = 1;
						}
						if (y1 + height > foreground.rows - 10 && ratt < 0.9)
						{
							templine[kk][12] = -1;
						}

						int flagb = 0;
						if ((x1 < 10 || x1 + width > 350) && height * width < c * d * aratio)
						{
							flagb = 1;
						}

						if (x - 0.1*c < 0 || x + 0.1*c > 360 || flagb == 1)
						{
							templine[kk][11] = -1;
						}

						if (x + c / 2 >= 360 && templine[kk][11] != -1 && templine[kk][12] != -1)
						{
							offset = (x - c / 2) - x1;
							newmid = x - offset;

							cam->imageToWorld(newmid * 2, (y1 + height) * 2, 0, X, Y);
							top3.x = X / 30 + 250;
							top3.y = Y / 30 + 250;

							cam->imageToWorld(newmid * 2,
								(b + d * 0.9) * 2, 0, X, Y);
							top1.x = X / 30 + 250;
							top1.y = Y / 30 + 250;

							cam->imageToWorld(newmid * 2,
								(b + d * 1.1) * 2, 0, X, Y);
							top2.x = X / 30 + 250;
							top2.y = Y / 30 + 250;

							templine[kk][0] = (int)(top1.x);
							templine[kk][1] = (int)(top1.y);
							templine[kk][2] = (int)(top2.x);
							templine[kk][3] = (int)(top2.y);
							templine[kk][4] = (int)(top3.x);
							templine[kk][5] = (int)(top3.y);

							flagn = 1;
						}

						else if (x - c / 2 <= 0 && templine[kk][11] != -1 && templine[kk][12] != -1)
						{
							offset = (x + c / 2) - (x1 + width);
							newmid = x - offset;

							cam->imageToWorld(newmid * 2, (y1 + height) * 2, 0, X, Y);
							top3.x = X / 30 + 250;
							top3.y = Y / 30 + 250;

							cam->imageToWorld(newmid * 2,
								(b + d * 0.9) * 2, 0, X, Y);
							top1.x = X / 30 + 250;
							top1.y = Y / 30 + 250;

							cam->imageToWorld(newmid * 2,
								(b + d * 1.1) * 2, 0, X, Y);
							top2.x = X / 30 + 250;
							top2.y = Y / 30 + 250;

							templine[kk][0] = (int)(top1.x);
							templine[kk][1] = (int)(top1.y);
							templine[kk][2] = (int)(top2.x);
							templine[kk][3] = (int)(top2.y);
							templine[kk][4] = (int)(top3.x);
							templine[kk][5] = (int)(top3.y);

							flagn = 1;

						}

						kk++;
					}
				}
			}
			else
			{
				for (int numI2 = 0; numI2 <= numI - 1; numI2++)
				{

					int numov = 0;
					double ttmax = 0;
					int yymax = 0;
					int flagf = 0;
					for (int numI3 = 0; numI3 <= numI - 1; numI3++)
					{
						incl = 0.8;

						if (numI2 != numI3)
						{	
							int bbrow = column[numI2];
							int x1 = bboxData.at<float>(bbrow, 2);
							int y1 = bboxData.at<float>(bbrow, 3);
							int width = bboxData.at<float>(bbrow, 4);
							int height = bboxData.at<float>(bbrow, 5);
							int xtp = bboxData.at<float>(bbrow, 8);
							int ytp = bboxData.at<float>(bbrow, 9);

							int bbrow2 = column[numI3];
							int x2 = bboxData.at<float>(bbrow2, 2);
							int y2 = bboxData.at<float>(bbrow2, 3);
							int width2 = bboxData.at<float>(bbrow2, 4);
							int height2 = bboxData.at<float>(bbrow2, 5);
							tempv1 = colOverlap(Rect(x1, y1, width, height), Rect(x2, y2, width2, height2));

							if (tempv1 > 0.3)
							{
								double XX, YY, XX2, YY2, xx, yy, xx2, yy2;
								double at, bt, ct, dt;

								cam->imageToWorld((x1 + width / 2) * 2, (y1 + height) * 2, 0, XX, YY);
								cam->worldToImage(XX, YY, 1800, xx, yy);
								xx = xx / 2;
								yy = yy / 2;
								cam->worldToImage(XX, YY, 0, xx2, yy2);
								xx2 = xx2 / 2;
								yy2 = yy2 / 2;

								at = xx2 - abs(yy2 - yy)*0.35*0.5;
								bt = yy;
								ct = abs(yy2 - yy)*0.35;
								dt = yy2 - yy;

								if (y1 + height < y2 + height2)
								{
									cam->imageToWorld((x1 + width / 2) * 2, (y1 + height) * 2, 0, X, Y);
									top3.x = X / 30 + 250;
									top3.y = Y / 30 + 250;

									if (top3.x >= 0 && top3.x <= 1000 && top3.y >= 0 && top3.y <= 1000)
									{										
										double kp, bp;
										kp = (vp.y - ytp) / (vp.x - xtp);
										bp = ytp - kp * xtp;

										int xtest, ytest;

										xtest = xtp;
										ytest = rctMap.at<Vec4f>(top3.x, top3.y)[1] + rctMap.at<Vec4f>(top3.x, top3.y)[3];

										if (ytest <= foreground.rows && ytest >= 0 && xtest <= foreground.cols && xtest >= 0)
										{
											uchar testf;
											for (double incll = 0.1; incll <= 1; incll = incll + 0.01)
											{
												xtest = xtp;
												ytest = rctMap.at<Vec4f>(top3.x, top3.y)[1] + rctMap.at<Vec4f>(top3.x, top3.y)[3] * incll;
												if (ytest <= foreground.rows && ytest >= 0 && xtest <= foreground.cols && xtest >= 0)
												{
													if (foreground2.at<uchar>(ytest, xtest) == 0)
													{
														continue;
													}
													else
													{
														testf = foreground2.at<uchar>(ytest, xtest);
														break;
													}
												}
											}

											for (double incl2 = 0.6; incl2 <= 1.2; incl2 = incl2 + 0.01)
											{
												xtest = xtp;
												ytest = rctMap.at<Vec4f>(top3.x, top3.y)[1] + rctMap.at<Vec4f>(top3.x, top3.y)[3] * incl2;

												if (ytest <= foreground.rows && ytest >= 0 && xtest <= foreground.cols && xtest >= 0)
												{
													if ((foreground2.at<uchar>(ytest, xtest) != testf) && (foreground2.at<uchar>(ytest, xtest) != 0))
													{
														flagf += 1;
													}
													if (ytest > foreground.rows || ytest < 0 || xtest > foreground.cols || xtest < 0 || incl > 1.2)
													{
														break;
													}
												}
												else
												{
													break;
												}

												if (flagf > 0)
												{
													break;
												}
											}

											if (flagf > 0)
											{
												int tfo = 0;
												if (tempv1 > ttmax)
												{
													ttmax = tempv1;

													for (double incl3 = 1; incl3 <= 100; incl3 = incl3 + 0.1)
													{
														xtest = xtp;
														ytest = rctMap.at<Vec4f>(top3.x, top3.y)[1] + rctMap.at<Vec4f>(top3.x, top3.y)[3] * incl3;
														tfo = ytest;
														if (ytest > foreground.rows || ytest < 0 || xtest > foreground.cols || xtest < 0 || incl > 1.2)
														{
															break;
														}
														if (foreground2.at<uchar>(ytest, xtest) == 0 || tfo >= y2 + height2)
														{
															break;
														}
													}
													if (y2 + height2 * 0.8 < y1 + height * 1.05)
													{
														yymax = int(y2 + height2);
													}
													else
													{
														yymax = int(y2 + height2 * 0.8);
													}
												}
												
												numov += 1;
											}
										}

									}

								}

							}
						}
					}
					if (numov == 0)
					{
						int bbrow = column[numI2];
						int x1 = bboxData.at<float>(bbrow, 2);
						int y1 = bboxData.at<float>(bbrow, 3);
						int width = bboxData.at<float>(bbrow, 4);
						int height = bboxData.at<float>(bbrow, 5);
						float kt = bboxData.at<float>(bbrow, 6);
						float bt = bboxData.at<float>(bbrow, 7);
						int xtp = bboxData.at<float>(bbrow, 8);
						int ytp = bboxData.at<float>(bbrow, 9);

						flagn = 0;
						
						cam->imageToWorld((x1 + width / 2) * 2, (y1 + height) * 2, 0, X, Y);
						top3.x = X / 30 + 250;
						top3.y = Y / 30 + 250;

						cam->imageToWorld((x1 + 2 + width / 2) * 2, (y1 + height) * 2, 0, sX, sY);
						SX = sX / 30 + 250;
						SY = sY / 30 + 250;

						sigk = abs(sqrt((top3.x - SX)*(top3.x - SX)
							+ (top3.y - SY)*(top3.y - SY)));

						if (top3.x >= 0 && top3.x <= 1000 && top3.y >= 0 && top3.y <= 1000)
						{
							X = (top3.x - 250) * 30; 
							Y = (top3.y - 250) * 30;
							cam->worldToImage(X, Y, 0, x, y);
							x = x / 2;
							y = y / 2;

							X = (top3.x - 250) * 30; 
							Y = (top3.y - 250) * 30;
							cam->worldToImage(X, Y, PHeight, xt, yt);
							xt = xt / 2;
							yt = yt / 2;

							a = x - abs(y - yt)*0.35*0.5;
							b = yt;
							c = abs(y - yt)*0.35;
							d = y - yt;
							e = a + c;
							f = b + d;

							if ((vp.x - xtp) != 0)
							{
								double ratt = height / d;

								if (y1 + height > foreground.rows - 10)
								{
									if (ratt < 0.9)
									{
										double kp, bp;
										kp = (vp.y - ytp) / (vp.x - xtp);
										bp = ytp - kp * xtp;

										cam->imageToWorld(((b + d * upr - bp) / kp) * 2,
											(b + d * upr) * 2, 0, X, Y);
									}
									else
									{
										cam->imageToWorld(((b + d * upr - bt) / kt) * 2,
											(b + d * upr) * 2, 0, X, Y);
									}
								}
								else
								{
									cam->imageToWorld(((b + d * upr - bt) / kt) * 2,
										(b + d * upr) * 2, 0, X, Y);
								}
								top1.x = X / 30 + 250;
								top1.y = Y / 30 + 250;

								if (y1 + height > foreground.rows - 10)
								{
									if (ratt < 0.9)
									{
										double kp, bp;
										kp = (vp.y - ytp) / (vp.x - xtp);
										bp = ytp - kp * xtp;

										cam->imageToWorld(((camy - bp) / kp) * 2,
											camy * 2, 0, X, Y);
									}
									else
									{
										cam->imageToWorld(((camy - bt) / kt) * 2,
											camy * 2, 0, X, Y);
									}

								}
								else
								{
									cam->imageToWorld(((b + d * 1.1 - bt) / kt) * 2,
										(b + d * 1.1) * 2, 0, X, Y);
								}
								top2.x = X / 30 + 250;
								top2.y = Y / 30 + 250;

								if (y1 + height > foreground.rows - 10 && ratt < 0.9)
								{
									double kp, bp;
									kp = (vp.y - ytp) / (vp.x - xtp);
									bp = ytp - kp * xtp;

									cam->imageToWorld(((b + d - bp) / kp) * 2,
										(b + d) * 2, 0, X, Y);
									top3.x = X / 30 + 250;
									top3.y = Y / 30 + 250;
								}
								else
								{
									cam->imageToWorld(((b + d - bt) / kt) * 2,
										(b + d) * 2, 0, X, Y);
									top3.x = X / 30 + 250;
									top3.y = Y / 30 + 250;
								}

								templine.push_back(vector<int>(15, 0));
								templine[kk][0] = (int)(top1.x);
								templine[kk][1] = (int)(top1.y);
								templine[kk][2] = (int)(top2.x);
								templine[kk][3] = (int)(top2.y);
								templine[kk][4] = (int)(top3.x);
								templine[kk][5] = (int)(top3.y);
								templine[kk][6] = (int)(width);
								templine[kk][7] = -1;
								templine[kk][9] = (int)(k);
								templine[kk][10] = 0;
								templine[kk][11] = 0;
								templine[kk][12] = 0;
								templine[kk][13] = 0;
								if (sigk >= 1)
								{
									templine[kk][14] = sigk * sigk;
								}
								else
								{
									templine[kk][14] = 1;
								}
								if (y1 + height > foreground.rows - 10 && ratt < 0.9)
								{
									templine[kk][12] = -1;
								}

								int flagb = 0;
								if ((x1 < 10 || x1 + width > 350) && height * width < c * d * aratio)
								{
									flagb = 1;
								}

								if (x - 0.1*c < 0 || x + 0.1*c > 360 || flagb == 1)
								{
									templine[kk][11] = -1;
								}

								if (x + c / 2 >= 360 && templine[kk][11] != -1 && templine[kk][12] != -1)
								{
									offset = (x - c / 2) - x1;
									newmid = x - offset;

									cam->imageToWorld(newmid * 2, (y1 + height) * 2, 0, X, Y);
									top3.x = X / 30 + 250;
									top3.y = Y / 30 + 250;

									cam->imageToWorld(newmid * 2,
										(b + d * 0.9) * 2, 0, X, Y);
									top1.x = X / 30 + 250;
									top1.y = Y / 30 + 250;

									cam->imageToWorld(newmid * 2,
										(b + d * 1.1) * 2, 0, X, Y);
									top2.x = X / 30 + 250;
									top2.y = Y / 30 + 250;

									templine[kk][0] = (int)(top1.x);
									templine[kk][1] = (int)(top1.y);
									templine[kk][2] = (int)(top2.x);
									templine[kk][3] = (int)(top2.y);
									templine[kk][4] = (int)(top3.x);
									templine[kk][5] = (int)(top3.y);

									flagn = 1;
								}

								else if (x - c / 2 <= 0 && templine[kk][11] != -1 && templine[kk][12] != -1)
								{
									offset = (x + c / 2) - (x1 + width);
									newmid = x - offset;

									cam->imageToWorld(newmid * 2, (y1 + height) * 2, 0, X, Y);
									top3.x = X / 30 + 250;
									top3.y = Y / 30 + 250;

									cam->imageToWorld(newmid * 2,
										(b + d * 0.9) * 2, 0, X, Y);
									top1.x = X / 30 + 250;
									top1.y = Y / 30 + 250;

									cam->imageToWorld(newmid * 2,
										(b + d * 1.1) * 2, 0, X, Y);
									top2.x = X / 30 + 250;
									top2.y = Y / 30 + 250;

									templine[kk][0] = (int)(top1.x);
									templine[kk][1] = (int)(top1.y);
									templine[kk][2] = (int)(top2.x);
									templine[kk][3] = (int)(top2.y);
									templine[kk][4] = (int)(top3.x);
									templine[kk][5] = (int)(top3.y);

									flagn = 1;
								}

								if (y1 + height >= foreground.rows - 10 && ratt < 0.9)
								{
									if (templine[kk][11] != -1 && flagn == 0)
									{
										double kp, bp;
										kp = (vp.y - ytp) / (vp.x - xtp);
										bp = ytp - kp * xtp;
									}

								}
								kk++;
							}
							else
							{
								double ratt = height / d;

								if (y1 + height > foreground.rows - 10 && ratt < 0.9)
								{
									cam->imageToWorld(x * 2,
										(b + d * upr) * 2, 0, X, Y);
								}
								else
								{
									cam->imageToWorld(((b + d * upr - bt) / kt) * 2,
										(b + d * upr) * 2, 0, X, Y);
								}
								top1.x = X / 30 + 250;
								top1.y = Y / 30 + 250;

								if (y1 + height > foreground.rows - 10)
								{
									if (ratt < 0.9)
									{
										cam->imageToWorld(x * 2,
											camy * 2, 0, X, Y);
									}
									else
									{
										cam->imageToWorld(x * 2,
											camy * 2, 0, X, Y);
									}
								}
								else
								{
									cam->imageToWorld(((b + d * 1.1 - bt) / kt) * 2,
										(b + d * 1.1) * 2, 0, X, Y);
								}
								top2.x = X / 30 + 250;
								top2.y = Y / 30 + 250;

								if (y1 + height > foreground.rows - 10 && ratt < 0.9)
								{
									cam->imageToWorld(x * 2,
										(b + d) * 2, 0, X, Y);
									top3.x = X / 30 + 250;
									top3.y = Y / 30 + 250;
								}
								else
								{
									cam->imageToWorld(((b + d - bt) / kt) * 2,
										(b + d) * 2, 0, X, Y);
								}

								templine.push_back(vector<int>(15, 0));
								templine[kk][0] = (int)(top1.x);
								templine[kk][1] = (int)(top1.y);
								templine[kk][2] = (int)(top2.x);
								templine[kk][3] = (int)(top2.y);
								templine[kk][4] = (int)(top3.x);
								templine[kk][5] = (int)(top3.y);
								templine[kk][6] = (int)(width);
								templine[kk][7] = -1;
								templine[kk][9] = (int)(k);
								templine[kk][10] = 0;
								templine[kk][11] = 0;
								templine[kk][12] = 0;
								templine[kk][13] = 0; 
								if (sigk >= 1)
								{
									templine[kk][14] = sigk * sigk;
								}
								else
								{
									templine[kk][14] = 1;
								}
								if (y1 + height > foreground.rows - 10 && ratt < 0.9)
								{
									templine[kk][12] = -1;
								}

								int flagb = 0;
								if ((x1 < 10 || x1 + width > 350) && height * width < c * d * aratio)
								{
									flagb = 1;
								}

								if (x - 0.1*c < 0 || x + 0.1*c > 360 || flagb == 1)
								{
									templine[kk][11] = -1;
								}

								if (x + c / 2 >= 360 && templine[kk][11] != -1 && templine[kk][12] != -1)
								{
									offset = (x - c / 2) - x1;
									newmid = x - offset;

									cam->imageToWorld(newmid * 2, (y1 + height) * 2, 0, X, Y);
									top3.x = X / 30 + 250;
									top3.y = Y / 30 + 250;

									cam->imageToWorld(newmid * 2,
										(b + d * 0.9) * 2, 0, X, Y);
									top1.x = X / 30 + 250;
									top1.y = Y / 30 + 250;

									cam->imageToWorld(newmid * 2,
										(b + d * 1.1) * 2, 0, X, Y);
									top2.x = X / 30 + 250;
									top2.y = Y / 30 + 250;

									templine[kk][0] = (int)(top1.x);
									templine[kk][1] = (int)(top1.y);
									templine[kk][2] = (int)(top2.x);
									templine[kk][3] = (int)(top2.y);
									templine[kk][4] = (int)(top3.x);
									templine[kk][5] = (int)(top3.y);

									flagn = 1; 
								}

								else if (x - c / 2 <= 0 && templine[kk][11] != -1 && templine[kk][12] != -1)
								{
									offset = (x + c / 2) - (x1 + width);
									newmid = x - offset;

									cam->imageToWorld(newmid * 2, (y1 + height) * 2, 0, X, Y);
									top3.x = X / 30 + 250;
									top3.y = Y / 30 + 250;

									cam->imageToWorld(newmid * 2,
										(b + d * 0.9) * 2, 0, X, Y);
									top1.x = X / 30 + 250;
									top1.y = Y / 30 + 250;

									cam->imageToWorld(newmid * 2,
										(b + d * 1.1) * 2, 0, X, Y);
									top2.x = X / 30 + 250;
									top2.y = Y / 30 + 250;

									templine[kk][0] = (int)(top1.x);
									templine[kk][1] = (int)(top1.y);
									templine[kk][2] = (int)(top2.x);
									templine[kk][3] = (int)(top2.y);
									templine[kk][4] = (int)(top3.x);
									templine[kk][5] = (int)(top3.y);

									flagn = 1; 
								}
								kk++;
							}
						}
					}
					else
					{
						int bbrow = column[numI2];
						int x1 = bboxData.at<float>(bbrow, 2);
						int y1 = bboxData.at<float>(bbrow, 3);
						int width = bboxData.at<float>(bbrow, 4);
						int height = bboxData.at<float>(bbrow, 5);
						int xtp = bboxData.at<float>(bbrow, 8);
						int ytp = bboxData.at<float>(bbrow, 9);

						double kp, bp, yb, xb;
						if (vp.x - xtp != 0)
						{
							kp = (vp.y - ytp) / (vp.x - xtp);
							bp = ytp - kp * xtp;
							yb = y1 + height;
							xb = (yb - bp) / kp;

							cam->imageToWorld(xb * 2, yb * 2, 0, X, Y);
							top3.x = X / 30 + 250;
							top3.y = Y / 30 + 250;

							cam->imageToWorld(((rctMap.at<Vec4f>(top3.x, top3.y)[1] + rctMap.at<Vec4f>(top3.x, top3.y)[3] * upr - bp) / kp) * 2,
								(rctMap.at<Vec4f>(top3.x, top3.y)[1] + rctMap.at<Vec4f>(top3.x, top3.y)[3] * upr) * 2, 0, X, Y);
							top1.x = X / 30 + 250;
							top1.y = Y / 30 + 250;

							cam->imageToWorld(((yymax - bp) / kp) * 2,
								yymax * 2, 0, X, Y);
							top2.x = X / 30 + 250;
							top2.y = Y / 30 + 250;

							templine.push_back(vector<int>(14, 0));
							templine[kk][0] = (int)(top1.x);
							templine[kk][1] = (int)(top1.y);
							templine[kk][2] = (int)(top2.x);
							templine[kk][3] = (int)(top2.y);
							templine[kk][4] = (int)(top3.x);
							templine[kk][5] = (int)(top3.y);
							templine[kk][6] = (int)(width);
							templine[kk][7] = -1;
							templine[kk][9] = (int)(k);
							templine[kk][10] = 0;
							templine[kk][11] = 0;
							templine[kk][12] = -1;
							templine[kk][13] = 0;

							X = (top3.x - 250) * 30;
							Y = (top3.y - 250) * 30;
							cam->worldToImage(X, Y, 0, x, y);
							x = x / 2;
							y = y / 2;

							X = (top3.x - 250) * 30; 
							Y = (top3.y - 250) * 30;
							cam->worldToImage(X, Y, PHeight, xt, yt);
							xt = xt / 2;
							yt = yt / 2;

							a = x - abs(y - yt)*0.35*0.5;
							b = yt;
							c = abs(y - yt)*0.35;
							d = y - yt;
							e = a + c;
							f = b + d;

							int flagb = 0;
							if ((x1 < 10 || x1 + width > 350) && height * width < c * d * aratio)
							{
								flagb = 1;
							}

							if (x - 0.1*c < 0 || x + 0.1*c > 360 || flagb == 1)
							{
								templine[kk][11] = -1;
							}
							kk++;
						}

						else
						{
							cam->imageToWorld(xtp * 2, (y1 + height) * 2, 0, X, Y);
							top3.x = X / 30 + 250;
							top3.y = Y / 30 + 250;

							cam->imageToWorld(xtp * 2,
								(rctMap.at<Vec4f>(top3.x, top3.y)[1] + rctMap.at<Vec4f>(top3.x, top3.y)[3] * upr) * 2, 0, X, Y);
							top1.x = X / 30 + 250;
							top1.y = Y / 30 + 250;

							cam->imageToWorld(xtp * 2, yymax * 2, 0, X, Y);
							top2.x = X / 30 + 250;
							top2.y = Y / 30 + 250;


							templine.push_back(vector<int>(14, 0));
							templine[kk][0] = (int)(top1.x);
							templine[kk][1] = (int)(top1.y);
							templine[kk][2] = (int)(top2.x);
							templine[kk][3] = (int)(top2.y);
							templine[kk][4] = (int)(top3.x);
							templine[kk][5] = (int)(top3.y);
							templine[kk][6] = (int)(width);
							templine[kk][7] = -1;
							templine[kk][9] = (int)(k);
							templine[kk][10] = 0;
							templine[kk][11] = 0;
							templine[kk][12] = -1;
							templine[kk][13] = 0;

							X = (top3.x - 250) * 30;
							Y = (top3.y - 250) * 30;
							cam->worldToImage(X, Y, PHeight, xt, yt);
							xt = xt / 2;
							yt = yt / 2;

							cam->worldToImage(X, Y, 0, x, y);
							x = x / 2;
							y = y / 2;

							a = x - abs(y - yt)*0.35*0.5;
							b = yt;
							c = abs(y - yt)*0.35;
							d = y - yt;
							e = a + c;
							f = b + d;

							int flagb = 0;
							if ((x1 < 10 || x1 + width > 350) && height * width < c * d * aratio)
							{
								flagb = 1;
							}

							if (x - 0.1*c < 0 || x + 0.1*c > 360 || flagb == 1)
							{
								templine[kk][11] = -1;
							}
							kk++;
						}
					}
				}
			}
			line.push_back(templine);
		}


		return numI;

	}

	bool isLineSegmentCross(const Point &P1, const Point &P2, const Point &Q1, const Point &Q2)
	{
		if (
			((Q1.x - P1.x)*(Q1.y - Q2.y) - (Q1.y - P1.y)*(Q1.x - Q2.x)) * ((Q1.x - P2.x)*(Q1.y - Q2.y) - (Q1.y - P2.y)*(Q1.x - Q2.x)) < 0 &&
			((P1.x - Q1.x)*(P1.y - P2.y) - (P1.y - Q1.y)*(P1.x - P2.x)) * ((P1.x - Q2.x)*(P1.y - P2.y) - (P1.y - Q2.y)*(P1.x - P2.x)) < 0
			)
			return true;
		else
			return false;
	}

	bool IsRectCross(const Point &p1, const Point &p2, const Point &q1, const Point &q2)
	{
		bool ret = min(p1.x, p2.x) <= max(q1.x, q2.x) &&
			min(q1.x, q2.x) <= max(p1.x, p2.x) &&
			min(p1.y, p2.y) <= max(q1.y, q2.y) &&
			min(q1.y, q2.y) <= max(p1.y, p2.y);
		return ret;
	}

	Point findLineCross(vector < vector<int> >  lines)
	{
		if (lines.size() == 1)
			return Point(0, 0);

		vector<Point> pt;
		Point pta;
		vector<Point> pts;
		vector<double>A;
		vector<double>B;
		vector<double>C;
		double A1, B1, C1, A2, B2, C2, AB;

		A1 = 0; B1 = 0; C1 = 0; A2 = 0; B2 = 0; C2 = 0;

		for (int kk = 0; kk < lines.size(); kk++)
		{
			A.push_back(lines[kk][3] - lines[kk][1]);
			B.push_back(lines[kk][0] - lines[kk][2]);
			C.push_back(lines[kk][2] * lines[kk][1] - lines[kk][0] * lines[kk][3]);
		}

		for (int kk = 0; kk < lines.size(); kk++)
		{
			AB = A[kk] * A[kk] + B[kk] * B[kk];
			A1 += 2 * A[kk] * A[kk] / AB;
			B1 += 2 * A[kk] * B[kk] / AB;
			C1 += 2 * A[kk] * C[kk] / AB;
			A2 += 2 * A[kk] * B[kk] / AB;
			B2 += 2 * B[kk] * B[kk] / AB;
			C2 += 2 * B[kk] * C[kk] / AB;
		}
		pta.x = (C2*B1 - B2 * C1) / (B2*A1 - B1 * A2);
		pta.y = (C2*A1 - A2 * C1) / (B1*A2 - B2 * A1);

		return pta;
	}

private:


};



