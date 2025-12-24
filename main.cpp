/* It is the program of the article: PPM: A Boolean Optimizer for Data Association in Multi-View Pedestrian Detection
 * It is authored by Rui Qiu, Ming Xu, Yuyao Yan, Jeremy S. Smith, Yuchen Ling.
 * This program is a x64 program and developed by using Microsoft Visual Studio 2017 (v141) and opencv 2.4.13.
 * Ground truth is from https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/
 */

#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
// OpenCV C-API declarations (CvFont, cvLine, cvPutText, CV_RGB, etc.)
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/opencv.hpp> 
// #include <cv.h>
#include <opencv2/opencv.hpp> // 包含所有核心功能
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <cxcore.h>

//#include <highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <cvaux.h>
#include <opencv2/video/tracking.hpp> 
#include <opencv2/objdetect/objdetect.hpp>
#include "cameraModel.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <conio.h>
#include<vector>
// 使用相对路径包含 Petrick 头文件（避免重复目录和转义问题）
#include "header file/petrick.h"
#include<string>
#include <cassert>
#include "camera.h"

#ifdef _WIN32
// Map deprecated POSIX names to MSVC variants
#define getch  _getch
#define kbhit  _kbhit
#define itoa   _itoa
#endif

using namespace std;

using namespace cv;

#define PHeight 2000  //average height of a pedestrian
#define eval 0 
#define weight 1
#define canm 4 //number of cameras
#define Rge 13 //search range

const int N = 100;
int visit[N];
int mark[N];
int match[N][N];
int nd, ng;
int ansH = 0;

int shorttable = 0;
int countcolumn[400];

Scalar colorful[40] = { CV_RGB(255, 128, 128), CV_RGB(255, 255, 128), CV_RGB(128, 0, 64), CV_RGB(128, 0, 255), CV_RGB(128, 255, 255), CV_RGB(0, 128, 255), CV_RGB(255, 0, 128), CV_RGB(255, 128, 255),
CV_RGB(128, 255, 128), CV_RGB(128, 128, 255), CV_RGB(0, 64, 128), CV_RGB(0, 128, 128), CV_RGB(0, 255, 0), CV_RGB(255, 128, 64), CV_RGB(128, 64, 64), CV_RGB(255, 0, 0), CV_RGB(255, 255, 0),
CV_RGB(0,255,255), CV_RGB(0, 255, 0), CV_RGB(255, 255, 255),
CV_RGB(255, 128, 128), CV_RGB(255, 255, 128), CV_RGB(128, 0, 64), CV_RGB(128, 0, 255), CV_RGB(128, 255, 255), CV_RGB(0, 128, 255), CV_RGB(255, 0, 128), CV_RGB(255, 128, 255),
CV_RGB(128, 255, 128), CV_RGB(128, 128, 255), CV_RGB(0, 64, 128), CV_RGB(0, 128, 128), CV_RGB(0, 255, 0), CV_RGB(255, 128, 64), CV_RGB(128, 64, 64), CV_RGB(255, 0, 0), CV_RGB(255, 255, 0),
CV_RGB(64, 0, 64), CV_RGB(0, 255, 128), CV_RGB(255, 255, 255) };


// Minimal CvBlob definition to replace external dependency
typedef struct CvBlob {
    double x;
    double y;
    double w;
    double h;
} CvBlob;

void PTKmethod(vector<Camera*> camera, Mat* st, int num, vector<int>& result);
void QMmap(CvRect rect, Mat* pFrame, int i);
int RemoveDuplates(Mat* A, Mat*B, Mat* fmask);
int removeRow(bool a[], bool b[], int num);
void showtable(int totalNum, int num, bool state[], bool** t, vector<Camera*> camera);
void drawDashRect(CvArr* img, int linelength, int dashlength, CvBlob* blob, CvScalar color, int thickness);
float bbOverlap(Rect box1, Rect box2);
int dfs(int i);
int Hungary();
Point findLineCross2(vector < vector<int> >  lines);
double determinant(double v1, double v2, double v3, double v4);
bool intersect(Point aa, Point bb, Point cc, Point dd);
void Nrec(Mat frm, vector<Camera*> camera, Point xcb, Point vr, int k, const cv::Scalar &color);

typedef struct Measure
{
	double prob;
	CvRect R1;
	CvRect R2;
	CvRect R3;
	IplImage* cmask[3];
	int state = 0;
	Point p;
};

Point GetFoot(const Point &pt, const Point &begin, const Point &end);

//Intersection point calculation
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
		if (lines[kk][12] != -1)
		{
			AB = A[kk] * A[kk] + B[kk] * B[kk];
			A1 += 2 * A[kk] * A[kk] / (AB * lines[kk][14]);
			B1 += 2 * A[kk] * B[kk] / (AB * lines[kk][14]);
			C1 += 2 * A[kk] * C[kk] / (AB * lines[kk][14]);
			A2 += 2 * A[kk] * B[kk] / (AB * lines[kk][14]);
			B2 += 2 * B[kk] * B[kk] / (AB * lines[kk][14]);
			C2 += 2 * B[kk] * C[kk] / (AB * lines[kk][14]);
		}
		else
		{
			AB = A[kk] * A[kk] + B[kk] * B[kk];
			A1 += weight * A[kk] * A[kk] / AB;
			B1 += weight * A[kk] * B[kk] / AB;
			C1 += weight * A[kk] * C[kk] / AB;
			A2 += weight * A[kk] * B[kk] / AB;
			B2 += weight * B[kk] * B[kk] / AB;
			C2 += weight * B[kk] * C[kk] / AB;
		}
	}
	pta.x = (C2*B1 - B2 * C1) / (B2*A1 - B1 * A2);
	pta.y = (C2*A1 - A2 * C1) / (B1*A2 - B2 * A1);

	return pta;
}


int main(int argc, const char * argv[])
{

	cv::Rect select_ROI;

    //EPFL Terrace Ground truth
    ifstream infile;
    infile.open(".\\PPM_code\\gt_terrace.txt"); 
    assert(infile.is_open());   

	string input_s;
	getline(infile, input_s);
	getline(infile, input_s);
	getline(infile, input_s);

	int input_int;
	vector<int> gt_n;
	vector<Rect> gt;

    //Mask Scoring RCNN bounding box outputs
    fstream file1;
    file1.open(".\\PPM_code\\Terrace_ms_5000.txt");
    Mat bboxData = Mat::zeros(89562, 10, CV_32FC1);

    if (!file1.is_open()) {
        cerr << "[Warn] Could not open Terrace_ms_5000.txt. Falling back to mask-based detections." << endl;
    } else {
        cout << "[Info] Successfully opened Terrace_ms_5000.txt, reading detection data..." << endl;
        for (int i = 0; i < 89562; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                if (!(file1 >> bboxData.at<float>(i, j))) {
                    cerr << "[Warn] Unexpected end of Terrace_ms_5000.txt at row " << i << ", col " << j << ". Some detections may be missing." << endl;
                    break;
                }
            }
        }
        cout << "[Info] Detection data loaded successfully." << endl;
    }

	char ch;
	int fstep = 0;

	vector<Camera*> camera(canm);
    //EPFL Terrace camera parameters, EPFL Terrace video can be downloaded from https://www.epfl.ch/labs/cvlab
    camera[0] = new Camera(".\\PPM_code\\Terrace\\terrace1-c0.avi");
	camera[0]->cam->setExtrinsic(-4.8441913843e+03, 5.5109448682e+02, 4.9667438357e+03, 1.9007833770e+00, 4.9730769727e-01, 1.8415452559e-01);
	camera[0]->cam->setGeometry(720, 576, 576, 576, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02);
	camera[0]->cam->setIntrinsic(20.161920, 5.720865e-04, 366.514507, 305.832552, 1);
	camera[0]->cam->internalInit();

    camera[1] = new Camera(".\\PPM_code\\Terrace\\terrace1-c1.avi");
	camera[1]->cam->setExtrinsic(-65.433635, 1594.811988, 2113.640844, 1.9347282363e+00, -7.0418616982e-01, -2.3783238362e-01);
	camera[1]->cam->setGeometry(720, 576, 576, 576, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02);
	camera[1]->cam->setIntrinsic(19.529144, 5.184242e-04, 360.228130, 255.166919, 1);
	camera[1]->cam->internalInit();

    camera[2] = new Camera(".\\PPM_code\\Terrace\\terrace1-c2.avi");
	camera[2]->cam->setExtrinsic(1.9782813424e+03, -9.4027627332e+02, 1.2397750058e+04, -1.8289537286e+00, 3.7748154985e-01, 3.0218614321e+00);
	camera[2]->cam->setGeometry(720, 576, 576, 576, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02);
	camera[2]->cam->setIntrinsic(19.903218, 3.511557e-04, 355.506436, 241.205640, 1.0000000000e+00);
	camera[2]->cam->internalInit();

    camera[3] = new Camera(".\\PPM_code\\Terrace\\terrace1-c3.avi");
	camera[3]->cam->setExtrinsic(4.6737509054e+03, -2.5743341287e+01, 8.4155952460e+03, -1.8418460467e+00, -4.6728290805e-01, -3.0205552749e+00);   //view 3
	camera[3]->cam->setGeometry(720, 576, 576, 576, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02);
	camera[3]->cam->setIntrinsic(20.047015, 4.347668e-04, 349.154019, 245.786168, 1);
	camera[3]->cam->internalInit();

#pragma omp parallel for
	for (int i = 0; i < camera.size(); i++)
	{
		camera[i]->mapToTop();
	}

	int nFrmNum = 0;

	cv::namedWindow("C0", cv::WINDOW_NORMAL);
	cv::resizeWindow("C0", 400, 300);
	cv::moveWindow("C0", 0, 0);
	cv::namedWindow("C1", cv::WINDOW_NORMAL);
	cv::resizeWindow("C1", 400, 300);
	cv::moveWindow("C1", 450, 0);
	cv::namedWindow("C2", cv::WINDOW_NORMAL);
	cv::resizeWindow("C2", 400, 300);
	cv::moveWindow("C2", 0, 400);
	cv::namedWindow("C3", cv::WINDOW_NORMAL);
	cv::resizeWindow("C3", 400, 300);
	cv::moveWindow("C3", 450, 400);
	cv::namedWindow("top", cv::WINDOW_NORMAL);
	cv::resizeWindow("top", 800, 800);
	cv::moveWindow("top", 850, 0);

	//initial parameters to store the intermediate results
	int num = 0;
	static Mat st(1, 400, CV_8U, Scalar(0));
	static Measure mea[400];
	int ex = 0;

	Mat topCross(1000, 1000, CV_8UC3);
	Mat topCross2(1000, 1000, CV_8UC3);
	Mat topCross3(1000, 1000, CV_8UC3);
	Mat topCross4(1000, 1000, CV_8UC3);

	vector<int> result;

	float countHit = 0;
	int countdet = 0;
	int countGT = 0;
	float MODP = 0;
	float NMODP = 0;
	float NMODA = 0;
	float RECALL, PRECISION, TER, FSCORE = 0;
	int Nframe = 0;

	float eresult[4][5];
	for (int ng = 0; ng < 4; ng++)
	{
		for (int ng2 = 0; ng2 < 5; ng2++)
		{
			eresult[ng][ng2] = 0;
		}
	}


	double X, Y, Z = 0;
	double x, y, xt, yt;
	Point pl[4];

	Point LT(285, 290), RB(450, 460);

    //Terrace top view
    Mat topback = imread(".\\PPM_code\\top4.jpg");

	vector < vector<int> >  lvp;
	//vanish point calculation
	double cvp[40][2] = { {25, 272}, {7, 194}, {107, 224}, {251, 277}, {76, 121}, {196, 148},
	{350, 176}, {195, 101}, {264, 105}, {344, 110},
	{340, 276}, {160, 279}, {242, 221}, {349, 173}, {34, 211}, {187, 156}, {300, 118},
	{15, 144}, {110, 110}, {187, 95},
	{330, 269}, {66, 278}, {246, 239}, {346, 202}, {21, 189}, {200, 156}, {338, 129},
	{29, 103}, {139, 93}, {256, 83},
	{24, 271}, {20, 165}, {151, 190}, {332, 214}, {74, 116}, {191, 136}, {335, 155},
	{169, 89}, {252, 98}, {346, 105} };

	int kv = -1;
	for (int nvp = 0; nvp < 40; nvp++)
	{
		if (nvp % 10 == 0)
		{
			kv += 1;
			vector < vector<int> >  lvp;

			if (kv >= canm)
			{
				break;
			}
		}

		double vpx, vpy, vpx2, vpy2;
		double vpX, vpY;

		camera[kv]->cam->imageToWorld(cvp[nvp][0] * 2, cvp[nvp][1] * 2, 0, vpX, vpY);

		camera[kv]->cam->worldToImage(vpX, vpY, 2000, vpx, vpy);
		vpx = vpx / 2;
		vpy = vpy / 2;
		camera[kv]->cam->worldToImage(vpX, vpY, 0, vpx2, vpy2);
		vpx2 = vpx2 / 2;
		vpy2 = vpy2 / 2;

		lvp.push_back(vector<int>(4, 0));
		lvp[lvp.size() - 1][0] = (int)(vpx);
		lvp[lvp.size() - 1][1] = (int)(vpy);
		lvp[lvp.size() - 1][2] = (int)(vpx2);
		lvp[lvp.size() - 1][3] = (int)(vpy2);

		if (nvp == 9 || nvp == 19 || nvp == 29 || nvp == 39)
		{
			camera[kv]->vp.x = findLineCross2(lvp).x;
			camera[kv]->vp.y = findLineCross2(lvp).y;
		}
	}

	while (1)
	{

		for (int i = 0; i < camera.size(); i++)
			camera[i]->readNextFrame();

		nFrmNum = camera[0]->FrameNumber;
		gt_n.clear();
		for (int i_in = 0; i_in < 9; i_in++)
		{
			infile >> input_int;
			if (input_int >= 0)
				gt_n.push_back(input_int);
		}

		if (nFrmNum > 5000)
			break;

		cout << "Current Frame Number: " << nFrmNum << endl;

		if (nFrmNum % 25 == 0)
		{
			nd = 0;
			ng = 0;

			for (int i = 0; i < 400; i++)
			{
				camera[0]->p[i] = Point(0, 0);
			}
#pragma omp parallel for
			for (int i = 0; i < camera.size(); i++)
				if (nFrmNum >= 1)
					if (i == 0)
						camera[i]->maskfore0();
					else if (i == 1)
						camera[i]->maskfore1();
					else if (i == 2)
						camera[i]->maskfore2();
					else
						camera[i]->maskfore3();

			int flag;
			int flag1;
			int flag2;

			topCross.setTo(0);

			topCross += topback;

			//Area of Interest (AOT) in topview
			rectangle(topCross, Rect(Point(290, 285), Point(460, 450)), CV_RGB(255, 0, 0), 2);;
			Point a(285, 290), b(450, 290), c(450, 460), d(285, 460);

			for (int k = 0; k < camera.size(); k++)
			{
				X = (a.x - 250) * 30;
				Y = (a.y - 250) * 30;
				Z = 0;
				camera[k]->cam->worldToImage(X, Y, Z, x, y);
				pl[0].x = x / 2;
				pl[0].y = y / 2;

				X = (b.x - 250) * 30;
				Y = (b.y - 250) * 30;
				Z = 0;
				camera[k]->cam->worldToImage(X, Y, Z, x, y);
				pl[1].x = x / 2;
				pl[1].y = y / 2;

				X = (c.x - 250) * 30;
				Y = (c.y - 250) * 30;
				Z = 0;
				camera[k]->cam->worldToImage(X, Y, Z, x, y);
				pl[2].x = x / 2;
				pl[2].y = y / 2;

				X = (d.x - 250) * 30;
				Y = (d.y - 250) * 30;
				Z = 0;
				camera[k]->cam->worldToImage(X, Y, Z, x, y);
				pl[3].x = x / 2;
				pl[3].y = y / 2;

				line(camera[k]->frame, pl[0], pl[1], CV_RGB(255, 0, 0), 2);
				line(camera[k]->frame, pl[1], pl[2], CV_RGB(255, 0, 0), 2);
				line(camera[k]->frame, pl[2], pl[3], CV_RGB(255, 0, 0), 2);
				line(camera[k]->frame, pl[3], pl[0], CV_RGB(255, 0, 0), 2);
			}

			vector < vector<int> >  lines;
			vector < vector<int> >  lines2;
			vector < vector<int> >  lines3;
			lines2.clear();
			vector <Point>  points;
			vector <Point>  points2;
			vector <Point>  points3;
			int numI = 0;
			float tempc0, tempc1, tempc2, tempc3, numt;

			//torso line extraction
			for (int k = 0; k < camera.size(); k++)
			{
				camera[k]->line.clear();
				int lineCount = camera[k]->lineti(nFrmNum, bboxData, k, topCross, topCross2, topCross3);
				numI += lineCount;
				cout << "[Debug] Camera " << k << " generated " << lineCount << " lines" << endl;
			}
			cout << "[Debug] Total lines generated: " << numI << endl;

			points.clear();

			points3.clear();
			lines3.clear();
			for (int k = 0; k < camera.size(); k++)
			{

				if (camera[k]->line.size() > 0)
				{
					for (int i = 0; i < camera[k]->line[0].size(); i++)
					{
						lines3.push_back(camera[k]->line[0][i]);
					}
				}
			}


			//initial each line group
			int nGs[100][100];
			for (int ig = 0; ig < 100; ig++)
			{
				for (int jg = 0; jg < 100; jg++)
				{
					nGs[ig][jg] = -1;
				}
			}

			//initial optimal variables
			int recp[4][2];
			for (int ir = 0; ir <= 3; ir++)
			{
				recp[ir][0] = -1;
			}
			for (int ir2 = 0; ir2 <= 3; ir2++)
			{
				recp[ir2][1] = 100;
			}

			//grouping record
			int grp[20][5];
			for (int ng = 0; ng < 20; ng++)
			{
				for (int ng2 = 0; ng2 < 5; ng2++)
				{
					grp[ng][ng2] = -1;
				}
			}

			//find the intersection point of each line projection
			for (int k = 0; k < lines3.size(); k++)
			{
				int gpi = 0;

				for (int ng = 0; ng < 20; ng++)
				{
					for (int ng2 = 0; ng2 < 5; ng2++)
					{
						grp[ng][ng2] = -1;
					}
				}

				tempc0 = 0;
				tempc1 = 0;
				lines.clear();

				if (lines3[k][12] != -1 && lines3[k][11] != -1 && lines3[k][10] == 0)
				{
					lines.push_back(lines3[k]);

					for (int k1 = 0; k1 < lines3.size(); k1++)
					{
						int gpn = 0;
						vector < vector<int> > Gp;
						Gp.clear();
						points2.clear();
						if (lines3[k1][11] != -1 && lines3[k1][9] != lines3[k][9] && nGs[k][k1] == -1)
						{
							lines.push_back(lines3[k1]);
							if (intersect(Point(lines3[k][0], lines3[k][1]), Point(lines3[k][2], lines3[k][3]),
								Point(lines3[k1][0], lines3[k1][1]), Point(lines3[k1][2], lines3[k1][3])))
							{
								points2.push_back(findLineCross(lines));

								Point testp;
								Point testp2;

								if (lines3[k1][12] != -1)
								{
									testp = GetFoot(Point(lines3[k][4], lines3[k][5]), Point(lines3[k1][0], lines3[k1][1]),
										Point(lines3[k1][2], lines3[k1][3]));
									testp2 = GetFoot(Point(lines3[k1][4], lines3[k1][5]), Point(lines3[k][0], lines3[k][1]),
										Point(lines3[k][2], lines3[k][3]));
								}
								else
								{
									testp = GetFoot(Point(lines3[k][4], lines3[k][5]), Point(lines3[k1][0], lines3[k1][1]),
										Point(lines3[k1][2], lines3[k1][3]));
									testp2 = testp;
								}

								if (testp.x != -1 && testp2.x != -1)
								{
									if (lines3[k1][12] != -1)
									{
										tempc0 = (abs(sqrt((lines3[k][4] - testp.x)*(lines3[k][4] - testp.x)
											+ (lines3[k][5] - testp.y)*(lines3[k][5] - testp.y))) +
											abs(sqrt((lines3[k1][4] - testp2.x)*(lines3[k1][4] - testp2.x)
												+ (lines3[k1][5] - testp2.y)*(lines3[k1][5] - testp2.y)))) / 2;
									}
									else
									{
										tempc0 = abs(sqrt((lines3[k][4] - testp.x)*(lines3[k][4] - testp.x)
											+ (lines3[k][5] - testp.y)*(lines3[k][5] - testp.y)));
									}
									if (tempc0 < Rge)
									{
										for (int ir = 0; ir <= 3; ir++)
										{
											recp[ir][0] = -1;
										}
										for (int ir2 = 0; ir2 <= 3; ir2++)
										{
											recp[ir2][1] = 100;
										}

										Gp.push_back(lines3[k]);
										Gp[Gp.size() - 1][13] = k;
										gpn += 1;

										Gp.push_back(lines3[k1]);
										Gp[Gp.size() - 1][13] = k1;
										gpn += 1;

										for (int k2 = 0; k2 < lines3.size(); k2++)
										{
											if ((lines3[k2][11] != -1) && (lines3[k][9] != lines3[k2][9]) && (lines3[k1][9] != lines3[k2][9]) && (nGs[k][k2] == -1) && (nGs[k1][k2] == -1))
											{
												Point testp3 = GetFoot(points2[0], Point(lines3[k2][0], lines3[k2][1]),
													Point(lines3[k2][2], lines3[k2][3]));

												if (testp3.x != -1)
												{
													tempc1 = abs(sqrt((testp3.x - points2[0].x)*(testp3.x - points2[0].x)
														+ (testp3.y - points2[0].y)*(testp3.y - points2[0].y)));

													if (tempc1 < Rge)
													{
														if (tempc1 < recp[lines3[k2][9]][1])
														{
															recp[lines3[k2][9]][0] = k2;
															recp[lines3[k2][9]][1] = tempc1;
														}
													}
												}
											}
										}

										double nts = 2;
										for (int ir3 = 0; ir3 < camera.size(); ir3++)
										{
											if (recp[ir3][0] != -1)
											{
												Gp.push_back(lines3[recp[ir3][0]]);
												nts++;
											}
										}
										Point tep = findLineCross(Gp);

										double ntp = 0;
										for (int kts = 0; kts < camera.size(); kts++)
										{
											if (camera[kts]->topshow.at<uchar>(tep.x, tep.y) == 1)
											{
												ntp++;
											}
										}

										double nrt = nts / ntp;

										if (nrt > 0.3)
										{
											for (int ir3 = 0; ir3 < camera.size(); ir3++)
											{
												if (recp[ir3][0] != -1)
												{
													Gp.push_back(lines3[recp[ir3][0]]);
													Gp[Gp.size() - 1][13] = recp[ir3][0];
												}
											}

											for (int ck = 0; ck < Gp.size(); ck++)
											{
												for (int cnp = 0; cnp < 7; cnp++)
												{
													if (Gp[ck][9] == cnp)
													{
														grp[gpi][cnp] = Gp[ck][13];
														gpn += 1;
													}
												}
											}

											grp[gpi][4] = gpn;
											gpi += 1;
										}
									}
								}
							}
							lines.pop_back();
						}
						points2.pop_back();
					}
					lines.pop_back();

					int maxg = 0;
					int maxi = -1;
					for (int gnn = 0; gnn < gpi; gnn++)
					{
						if (grp[gnn][4] > maxg)
						{
							maxg = grp[gnn][4];
							maxi = gnn;
						}
					}
					vector < vector<int> > Gp2;
					if (maxi >= 0)
					{
						for (int gc = 0; gc < camera.size(); gc++)
						{
							if (grp[maxi][gc] != -1)
							{
								Gp2.push_back(lines3[grp[maxi][gc]]);

								lines3[grp[maxi][gc]][10] = 1; 

								for (int gc2 = gc - 1; gc2 >= 0; gc2--)
								{
									nGs[grp[maxi][gc]][grp[maxi][gc2]] = 1;
									nGs[grp[maxi][gc2]][grp[maxi][gc]] = 1;
								}
							}
						}
						points3.push_back(findLineCross(Gp2));
					}

					int flagx; 			
					if (maxi >= 0)
					{
						for (int gnn = 0; gnn < gpi; gnn++)
						{
							flagx = 0;
							vector < vector<int> >  lines4;
							if (grp[gnn][4] == maxg && gnn != maxi)
							{
								for (int gc = 0; gc < camera.size(); gc++)
								{
									if (grp[gnn][gc] != -1)
									{
										lines4.push_back(lines3[grp[gnn][gc]]);
										lines3[grp[gnn][gc]][10] = 1;
										for (int gc2 = gc - 1; gc2 >= 0; gc2--)
										{
											if (nGs[grp[gnn][gc]][grp[gnn][gc2]] == -1)
											{
												nGs[grp[gnn][gc]][grp[gnn][gc2]] = 1;
												nGs[grp[gnn][gc2]][grp[gnn][gc]] = 1;
											}
											else
											{
												flagx = 1;
												break;
											}
										}
										if (flagx == 1)
										{
											break;
										}
									}
								}
								if (lines4.size() > 0 && flagx == 0)
								{
									points3.push_back(findLineCross(lines4));
								}
							}
						}
					}



				}
			}

			//remaining parallel lines
			for (int k = 0; k < lines3.size(); k++)
			{
				if (lines3[k][10] == 0 && lines3[k][11] != -1 && lines3[k][12] != -1)
				{
					for (int k1 = 0; k1 < lines3.size(); k1++)
					{
						Point testp;
						Point testp2;

						if (lines3[k1][10] == 0 && lines3[k1][11] != -1 && nGs[k][k1] == -1 && lines3[k1][9] != lines3[k][9])
						{
							testp = GetFoot(Point(lines3[k][4], lines3[k][5]), Point(lines3[k1][0], lines3[k1][1]),
								Point(lines3[k1][2], lines3[k1][3]));
							testp2 = GetFoot(Point(lines3[k1][4], lines3[k1][5]), Point(lines3[k][0], lines3[k][1]),
								Point(lines3[k][2], lines3[k][3]));

							if (testp.x != -1 && testp2.x != -1)
							{
								if (lines3[k1][12] != -1)
								{
									tempc0 = (abs(sqrt((lines3[k][4] - testp.x)*(lines3[k][4] - testp.x)
										+ (lines3[k][5] - testp.y)*(lines3[k][5] - testp.y))) +
										abs(sqrt((lines3[k1][4] - testp2.x)*(lines3[k1][4] - testp2.x)
											+ (lines3[k1][5] - testp2.y)*(lines3[k1][5] - testp2.y)))) / 2;
								}

								else
								{
									tempc0 = sqrt((lines3[k][4] - testp.x)*(lines3[k][4] - testp.x)
										+ (lines3[k][5] - testp.y)*(lines3[k][5] - testp.y));
								}

								if (tempc0 < Rge)
								{
									double nts = 2;

									double ntp = 0;
									for (int kts = 0; kts < camera.size(); kts++)
									{
										if (camera[kts]->topshow.at<uchar>(Point((lines3[k][4] + lines3[k1][4]) / 2, (lines3[k][5] + lines3[k1][5]) / 2)) == 1)
										{
											ntp++;
										}
									}

									double nrt = nts / ntp;
									if (nrt > 0.3)
									{
										if (lines3[k1][12] != -1)
										{										
											points3.push_back(Point((lines3[k][4] + lines3[k1][4]) / 2, (lines3[k][5] + lines3[k1][5]) / 2));

											lines3[k][10] = 1;
											lines3[k1][10] = 1;
											nGs[k][k1] = 1;
											nGs[k1][k] = 1;
										}
										else
										{									
											points3.push_back(Point((lines3[k][4] + testp.x) / 2, (lines3[k][5] + testp.y) / 2));
											lines3[k][10] = 1;
											nGs[k][k1] = 1;
											nGs[k1][k] = 1;
										}
									}

								}


							}
						}
					}
				}
			}

			//intersection point between long lines
			vector < vector<int> >  lines5;
			for (int k = 0; k < lines3.size(); k++)
			{
				if (lines3[k][11] != -1 && lines3[k][12] == -1)
				{
					lines5.push_back(lines3[k]);

					for (int k1 = 0; k1 < lines3.size(); k1++)
					{
						if (lines3[k1][11] != -1 && lines3[k1][12] == -1 && nGs[k][k1] == -1 && lines3[k1][9] != lines3[k][9])
						{
							if (intersect(Point(lines3[k][0], lines3[k][1]), Point(lines3[k][2], lines3[k][3]),
								Point(lines3[k1][0], lines3[k1][1]), Point(lines3[k1][2], lines3[k1][3])))
							{
								lines5.push_back(lines3[k1]);

								double nts = 2;

								double ntp = 0;
								for (int kts = 0; kts < camera.size(); kts++)
								{
									if (camera[kts]->topshow.at<uchar>(findLineCross(lines5)) == 1)
									{
										ntp++;
									}
								}

								double nrt = nts / ntp;
								if (nrt > 0.3)
								{									
									points3.push_back(findLineCross(lines5));

									lines3[k][10] = 1;
									lines3[k1][10] = 1;
									nGs[k][k1] = 1;
									nGs[k1][k] = 1;
								}
								lines5.pop_back();
							}
						}
					}
					lines5.pop_back();
				}
			}

			//remaining short lines
			for (int k = 0; k < lines3.size(); k++)
			{
				if (lines3[k][10] == 0 && lines3[k][11] != -1 && lines3[k][12] != -1)
				{
					double nts = 1;

					double ntp = 0;
					for (int kts = 0; kts < camera.size(); kts++)
					{
						if (camera[kts]->topshow.at<uchar>(Point(lines3[k][4], lines3[k][5])) == 1)
						{
							ntp++;
						}
					}

					double nrt = nts / ntp;
					if (nrt > 0.3)
					{
						points3.push_back(Point(lines3[k][4], lines3[k][5]));
					}
				}
			}

			st.setTo(0);
			int num2 = 0;
			//check foreground ratio
			for (int i = 0; i < points3.size(); i++)
			{
				if (points3[i].x != 0 || points3[i].y != 0)
				{
					for (int k = 0; k < camera.size(); k++)
					{
						X = (points3[i].x - 250) * 30;
						Y = (points3[i].y - 250) * 30;
						Z = 0;
						camera[k]->cam->worldToImage(X, Y, Z, x, y);
						x = x / 2;
						y = y / 2;
						Z = PHeight;
						camera[k]->cam->worldToImage(X, Y, Z, xt, yt);
						xt = xt / 2;
						yt = yt / 2;

						if (points3[i].x >= 0 && points3[i].x <= 1000 && points3[i].y >= 0 && points3[i].y <= 1000)
						{
							if (camera[k]->top.at<uchar>(points3[i].x, points3[i].y) == 1)
							{
								st.at<uchar>(0, i) += pow(2, k);
								camera[k]->r[i] = Rect(camera[k]->rctMap.at<Vec4f>(points3[i].x, points3[i].y)[0],
									camera[k]->rctMap.at<Vec4f>(points3[i].x, points3[i].y)[1],
									camera[k]->rctMap.at<Vec4f>(points3[i].x, points3[i].y)[2],
									camera[k]->rctMap.at<Vec4f>(points3[i].x, points3[i].y)[3]);

								camera[k]->probM[i] = 1;

								camera[k]->prob[i] = camera[k]->dutyCycle(points3[i].x, points3[i].y);
								if (camera[k]->prob[i] < 0.1)
								{
									camera[k]->prob[i] = 0.1;
								}
								camera[k]->probB[i] = 1;
								camera[k]->probT[i] = 1;
							}
							else
							{
								camera[k]->probM[i] = 1;
								camera[k]->prob[i] = 1;
								camera[k]->probB[i] = 1;
								camera[k]->probT[i] = 1;
							}
						}

					}
					num2++;
				}
			}

			double cnmm;
			double probs[400];
			for (int i = 0; i < points3.size(); i++)
			{
				cnmm = camera.size();
				probs[i] = 1;
				for (int k = 0; k < camera.size(); k++)
				{
					probs[i] *= camera[k]->prob[i];
					if (camera[k]->prob[i] == 1)
						cnmm--;
				}
				probs[i] = pow(probs[i], 1.0 / cnmm);

				for (int k = 0; k < camera.size(); k++)
				{
					if ((points3[i].x != 0 || points3[i].y != 0) && probs[i] >= 0.3)
					{
						camera[k]->p[i] = points3[i];
					}
				}
			}


			result.clear();

			//Petrick’s method
			PTKmethod(camera, &st, num2, result);

			//draw bounding boxes in each view and points in top view
			bool flagrst = 0;
			for (int i = 0; i < num2; i++)
			{
				flagrst = 0;
				for (int j = 0; j < result.size(); j++)
				{
					if (i == result[j])
						flagrst = 1;
				}
				if (flagrst == 1)
				{
					circle(topCross, camera[0]->p[i], 4, colorful[i], -1);
					for (int ksr = 0; ksr < camera.size(); ksr++)
					{
						Nrec(camera[ksr]->frame, camera, camera[0]->p[i], camera[ksr]->vp, ksr, colorful[i]);
					}
				}
			}

			flip(topCross, topCross, 0);

			putText(topCross, "C0", Point(camera[0]->cam->mCposx / 30.0 + 200, 1000 - (camera[0]->cam->mCposy / 30.0 + 300)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 2, CV_AA);
			putText(topCross, "C1", Point(camera[1]->cam->mCposx / 30.0 + 250, 1000 - (camera[1]->cam->mCposy / 30.0 + 250)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 255, 0), 2, CV_AA);
			putText(topCross, "C2", Point(camera[2]->cam->mCposx / 30.0 + 250 - 25, 1000 - (camera[2]->cam->mCposy / 30.0 + 250 - 25)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0, 255, 255), 2, CV_AA);
			putText(topCross, "C3", Point(camera[3]->cam->mCposx / 30.0 + 250, 1000 - (camera[3]->cam->mCposy / 30.0 + 250)), cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0, 255, 0), 2, CV_AA);

			//frame number in top view
			char sz[10];
			itoa(nFrmNum, sz, 10);
			putText(topCross, sz, Point(141, 416), cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1, CV_AA);

			imshow("C0", camera[0]->frame); waitKey(1);
			imshow("C1", camera[1]->frame); waitKey(1);
			imshow("C2", camera[2]->frame); waitKey(1);
			imshow("C3", camera[3]->frame); waitKey(1);

			select_ROI = Rect(125, 375, 500, 500);
			Mat ROI = topCross(select_ROI);
			imshow("top", ROI); waitKey(1);

			cout << "Current Frame Number:" << "" << nFrmNum << endl;

			//evaluation
			if (eval == 0)
			{
				float overlap;
				float distance;
				bool flagdraw;
				bool flagmiss = 0;

				int miss_count = 0;
				int miss_countk = 0;

				float gdIOU = 0;
				float tempdistance = 1000;
				float frIOU = 0;
				float totIOU = 0; 
				int tNframe = 0;

				int canum, frHit = 0; 
				float tfrIOU = 0;

				memset(match, 0, sizeof(match));

				int ga, gb, gn;
				float wga, wgb, wdx, wdy;

				if (gt_n.size() > 0)
				{
					for (int gt_in = 0; gt_in < gt_n.size(); gt_in++)
					{
						ga = gt_n[gt_in] % 30 * 250 + 125 - 500;
						gb = gt_n[gt_in] / 30 * 250 + 125 - 1500;

						wga = gt_n[gt_in] % 30 * 250 + 125 - 500; 
						wgb = gt_n[gt_in] / 30 * 250 + 125 - 1500; 

						ga = ga / 30 + 250;
						gb = gb / 30 + 250;

						for (int i = 0; i < result.size(); i++)
						{

							wdx = (camera[0]->p[result[i]].x - 250) * 30;
							wdy = (camera[0]->p[result[i]].y - 250) * 30;

							distance = abs(sqrt((wdx - wga)*(wdx - wga)
								+ (wdy - wgb)*(wdy - wgb)));
							if (distance <= 500)
							{
								match[gt_in][i] = 1;
							}
						}
					}
				}
				ng = gt_n.size(); 
				nd = result.size();

				ansH = Hungary();

				if (gt_n.size() > 0)
				{
					for (int j = 0; j < ng; j++)
					{
						int detIdx = -1;
						for (int i = 0; i < nd; i++)
						{
							if (mark[i] == j)
							{
								detIdx = i;
								break;
							}
						}

						ga = gt_n[j] % 30 * 250 + 125 - 500;
						gb = gt_n[j] / 30 * 250 + 125 - 1500;
						ga = ga / 30 + 250;
						gb = gb / 30 + 250;

						if (detIdx != -1)
						{
							if ((ga >= LT.x - 5 && ga <= RB.x + 5 && gb >= LT.y - 5 && gb <= RB.y + 5) || (camera[0]->p[result[detIdx]].x >= LT.x - 5 && camera[0]->p[result[detIdx]].x <= RB.x + 5 && camera[0]->p[result[detIdx]].y >= LT.y - 5 && camera[0]->p[result[detIdx]].y <= RB.y + 5))
							{
								countGT++;  
								countdet++;
								countHit++;
								frHit++;							
								tNframe++;

								for (int k = 0; k < camera.size(); k++)
								{
									if (camera[k]->topshow.at<uchar>(ga, gb) == 1)
									{
										overlap = bbOverlap(Rect(camera[k]->rctMap.at<Vec4f>(ga, gb)[0], camera[k]->rctMap.at<Vec4f>(ga, gb)[1],
											camera[k]->rctMap.at<Vec4f>(ga, gb)[2], camera[k]->rctMap.at<Vec4f>(ga, gb)[3]),
											Rect(camera[k]->rctMap.at<Vec4f>(int(camera[k]->p[result[detIdx]].x), int(camera[k]->p[result[detIdx]].y))[0], camera[k]->rctMap.at<Vec4f>(int(camera[k]->p[result[detIdx]].x), int(camera[k]->p[result[detIdx]].y))[1],
												camera[k]->rctMap.at<Vec4f>(int(camera[k]->p[result[detIdx]].x), int(camera[k]->p[result[detIdx]].y))[2], camera[k]->rctMap.at<Vec4f>(int(camera[k]->p[result[detIdx]].x), int(camera[k]->p[result[detIdx]].y))[3]));

										gdIOU += overlap;
										canum += 1;
									}
								}

								tfrIOU = gdIOU / canum;
								totIOU += tfrIOU;

								canum = 0;
								tfrIOU = 0;
								gdIOU = 0;
							}
						}
						else 
						{
							if (ga >= LT.x + 5 && ga <= RB.x - 5 && gb >= LT.y + 5 && gb <= RB.y - 5)
							{
								countGT++; 
							}
						}
					}

					for (int i = 0; i < nd; i++)
					{
						if (mark[i] == -1)
						{
							if (camera[0]->p[result[i]].x >= LT.x + 5 && camera[0]->p[result[i]].x <= RB.x - 5 && camera[0]->p[result[i]].y >= LT.y + 5 && camera[0]->p[result[i]].y <= RB.y - 5)
							{
								countdet++; 
							}
						}
					}

					if (tNframe > 0)
					{
						Nframe++;
					}

					if (totIOU > 0)
					{
						MODP += totIOU / frHit;
					}
					totIOU = 0;
					frHit = 0;
				}
			}

			else
			{
				vector<int> result2;
				result2 = result;
				for (int kev = 0; kev < camera.size(); kev++)
				{
					result = result2;

					float overlap;
					float distance;
					bool flagdraw;
					bool flagmiss = 0;

					int miss_count = 0;
					int miss_countk = 0;

					float gdIOU = 0; 
					float tempdistance = 1000;
					float frIOU = 0;
					float totIOU = 0;
					int tNframe = 0;

					int canum, frHit = 0;
					float tfrIOU = 0;

					int flaga, flagb, flagaf, flagbf = 0;
					int ga, gb, gn;



					if (gt_n.size() > 0)
					{
						for (int gt_in = 0; gt_in < gt_n.size(); gt_in++)
						{

							flagaf = 0;
							flagbf = 0;

							flagdraw = 0;
							ga = gt_n[gt_in] % 30 * 250 + 125 - 500;
							gb = gt_n[gt_in] / 30 * 250 + 125 - 1500;

							ga = ga / 30 + 250;
							gb = gb / 30 + 250;


							flag1 = 0;
							for (int k = 0; k < camera.size(); k++)
							{
								if (k == kev)
								{
									if (camera[k]->topshow.at<uchar>(ga, gb) == 1)
										flag1++;
								}
							}

							if (flag1 >= 1)
							{
								int tnum = 0;
								float maxIOU = 0;

								for (int i = 0; i < result.size(); i++)
								{
									flaga = 0; 
									flagb = 0;
									if (result[i] != -1)
									{
										for (int k = 0; k < camera.size(); k++)
										{
											if (k == kev)
											{
												if (camera[k]->topshow.at<uchar>(ga, gb) == 1)
												{
													overlap = bbOverlap(Rect(camera[k]->rctMap.at<Vec4f>(ga, gb)[0], camera[k]->rctMap.at<Vec4f>(ga, gb)[1],
														camera[k]->rctMap.at<Vec4f>(ga, gb)[2], camera[k]->rctMap.at<Vec4f>(ga, gb)[3]),
														Rect(camera[k]->rctMap.at<Vec4f>(int(camera[0]->p[result[i]].x), int(camera[0]->p[result[i]].y))[0], camera[k]->rctMap.at<Vec4f>(int(camera[0]->p[result[i]].x), int(camera[0]->p[result[i]].y))[1],
															camera[k]->rctMap.at<Vec4f>(int(camera[0]->p[result[i]].x), int(camera[0]->p[result[i]].y))[2], camera[k]->rctMap.at<Vec4f>(int(camera[0]->p[result[i]].x), int(camera[0]->p[result[i]].y))[3]));

													gdIOU += overlap;
													canum += 1;
													flaga += 1;
													if (overlap < 0.5)
													{
														if ((camera[k]->rctMap.at<Vec4f>(ga, gb)[0] <= 2
															|| (camera[k]->rctMap.at<Vec4f>(ga, gb)[0] + camera[k]->rctMap.at<Vec4f>(ga, gb)[2]) >= camera[k]->frame.cols - 2)
															&& camera[k]->rctMap.at<Vec4f>(ga, gb)[2] / camera[k]->rctMap.at<Vec4f>(ga, gb)[3] < 0.34)
														{
															flagb += 1;
														}
													}
													else
													{
														flagb += 1;
													}
												}
											}
										}
										if (flagb != 0)
										{
											tfrIOU = gdIOU / canum;
											if (tfrIOU > maxIOU)
											{
												tnum = i;
												maxIOU = tfrIOU;
												flagaf = flaga;
												flagbf = flagb;
											}
											tfrIOU = 0;
											gdIOU = 0;
											canum = 0;
										}
										else
										{
											gdIOU = 0;
											canum = 0;
										}
									}
								}

								if (maxIOU)
								{
									if ((ga >= LT.x - 5 && ga <= RB.x + 5 && gb >= LT.y - 5 && gb <= RB.y + 5) || (camera[0]->p[result[tnum]].x >= LT.x - 5 &&
										camera[0]->p[result[tnum]].x <= RB.x + 5 && camera[0]->p[result[tnum]].y >= LT.y - 5 &&
										camera[0]->p[result[tnum]].y <= RB.y + 5))
									{
										eresult[kev][0]++;
										eresult[kev][2]++;
										if (flagaf == flagbf)
										{
											eresult[kev][1]++;
										}
										else
										{
											eresult[kev][1] += float(flagbf) / float(flagaf);
										}
										frHit++;						
										tNframe++;
										result[tnum] = -1;
										totIOU += maxIOU;
									}
								}
								else
								{
									if (camera[kev]->topshow.at<uchar>(ga, gb) == 1)
									{
										if (ga >= LT.x + 5 && ga <= RB.x - 5 && gb >= LT.y + 5 && gb <= RB.y - 5)
										{
											eresult[kev][0]++;
										}
									}
								}

							}

						}

						for (int i = 0; i < result.size(); i++)
						{
							if (result[i] != -1 && camera[kev]->topshow.at<uchar>(camera[0]->p[result[i]].x, camera[0]->p[result[i]].y) == 1)
							{
								if (camera[0]->p[result[i]].x >= LT.x + 5 && camera[0]->p[result[i]].x <= RB.x - 5
									&& camera[0]->p[result[i]].y >= LT.y + 5 && camera[0]->p[result[i]].y <= RB.y - 5)
								{
									eresult[kev][2]++;
								}
							}
						}

						if (tNframe > 0)
						{
							eresult[kev][4]++;
						}

						if (totIOU > 0)
						{
							eresult[kev][3] += totIOU / frHit;
						}
						totIOU = 0;
						frHit = 0;
					}
				}
			}
		}

		if (fstep == 1 && nFrmNum % 1 == 0) {
			ch = getch();
			if (ch == 13) fstep = 0;
			if (ch == 27) break;
		}
		else {
			if (kbhit()) {
				ch = getch();
				if (ch == 32) fstep = 1;
				else break;
			}
		}


	}

	NMODP = MODP / Nframe;
	TER = ((countGT - countHit) + (countdet - countHit)) / double(countGT);
	PRECISION = countHit / double(countdet);
	RECALL = countHit / double(countGT);
	NMODA = 1 - TER;
	FSCORE = 2 * PRECISION*RECALL / (PRECISION + RECALL);

	cout << endl << endl << endl;
	cout << "count GT: " << countGT << " " << "count detection: " << countdet << " " << "count matched: " << countHit << endl;
	cout << "FN " << countGT - countHit << endl;
	cout << "FN rate" << (countGT - countHit) / double(countGT) << endl;
	cout << "FP " << countdet - countHit << endl;
	cout << "FP rate" << (countdet - countHit) / double(countdet) << endl;
	cout << "MDR " << (countGT - countHit) / double(countGT) << endl;
	cout << "FDR " << (countdet - countHit) / double(countGT) << endl;
	cout << "TER " << ((countGT - countHit) + (countdet - countHit)) / double(countGT) << endl;
	cout << "PRE " << countHit / double(countdet) << endl;
	cout << "REC " << countHit / double(countGT) << endl;
	cout << "F-Score: " << FSCORE << endl;
	cout << "N-MODA: " << NMODA << endl;
	cout << "N-MODP: " << NMODP << endl;

	return 0;
}

void PTKmethod(vector<Camera*> camera, Mat* st, int num, vector<int>& result)
{
	int n = 0;
	CvFont font;
	char text[10];
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.5, 0.5, 0, 2);

	bool state[400];
	double prob[400];
	int camNum[400];

	shorttable = 0;

	cout << "\t\b\b\b\b" << "  ";

	for (int k = 0; k < camera.size(); k++)
	{
		cout << k + 1 << "F    " << k + 1 << "T    " << k + 1 << "B    " << k + 1 << "M    ";
	}
	cout << " JL" << endl;

	for (int i = 0; i < num; i++)
	{
		cout << "I" << i << "\t\b\b\b\b";
		prob[i] = 1;
		camNum[i] = camera.size();
		for (int k = 0; k < camera.size(); k++)
		{
			cout << fixed << setw(5) << setprecision(3) << camera[k]->prob[i] << " " << fixed << setw(5) << setprecision(3) <<
				camera[k]->probT[i] << " " << fixed << setw(5) << setprecision(3) << camera[k]->probB[i] << " "
				<< fixed << setw(5) << setprecision(3) << camera[k]->probM[i] << " ";
			prob[i] *= camera[k]->prob[i] * camera[k]->probT[i] * camera[k]->probB[i] * camera[k]->probM[i];
			if (camera[k]->prob[i] == 1)
				camNum[i]--;
		}
		prob[i] = pow(prob[i], 1.0 / camNum[i]);
		cout << " " << prob[i] << endl;
	}

	int inx = 0;
	while (inx < num)
	{
		if (prob[inx] < 0.3)
		{
			for (int j = inx; j <= num - 1; j++)
			{
				for (int m = 0; m < camera.size(); m++)
				{
					camera[m]->r[j] = camera[m]->r[j + 1];
					camera[m]->prob[j] = camera[m]->prob[j + 1];
					camera[m]->probB[j] = camera[m]->probB[j + 1];
					camera[m]->probT[j] = camera[m]->probT[j + 1];
					camera[m]->probM[j] = camera[m]->probM[j + 1];
					camera[m]->p[j] = camera[m]->p[j + 1];
					prob[j] = prob[j + 1];
					st->at<uchar>(0, j) = st->at<uchar>(0, j + 1);
				}
			}
			num--;
			inx--;
		}
		inx++;
	}

	cout << endl;
	cout << "Modified Table" << endl;
	cout << "\t\b\b\b\b" << "  ";

	for (int k = 0; k < camera.size(); k++)
	{
		cout << k + 1 << "F    " << k + 1 << "T    " << k + 1 << "B    " << k + 1 << "M    ";
	}
	cout << " JL" << endl;

	for (int i = 0; i < num; i++)
	{
		cout << "I" << i << "\t\b\b\b\b";
		prob[i] = 1;
		camNum[i] = camera.size();
		for (int k = 0; k < camera.size(); k++)
		{
			cout << fixed << setw(5) << setprecision(3) << camera[k]->prob[i] << " " << fixed << setw(5) << setprecision(3)
				<< camera[k]->probT[i] << " " << fixed << setw(5) << setprecision(3) << camera[k]->probB[i] << " "
				<< fixed << setw(5) << setprecision(3) << camera[k]->probM[i] << " ";
			prob[i] *= camera[k]->prob[i] * camera[k]->probT[i] * camera[k]->probB[i] * camera[k]->probM[i];//multiply
			if (camera[k]->prob[i] == 1)
				camNum[i]--;

		}
		prob[i] = pow(prob[i], 1.0 / camNum[i]);
		cout << " " << prob[i] << endl;
	}

	for (int i = 0; i < camera.size(); i++)
	{
		camera[i]->pF.setTo(0);
		camera[i]->T.setTo(0);
	}

	for (int i = 0; i < num; i++)
	{
		state[i] = 0;
		for (int k = 0; k < camera.size(); k++)
			//0x01
			if (st->at<uchar>(0, i) >> k & 0x01)
				QMmap(camera[k]->r[i], &camera[k]->pF, i);
	}

	for (int k = 0; k < camera.size(); k++)
		camera[k]->Tnum = RemoveDuplates(&camera[k]->pF, &camera[k]->T, &camera[k]->foreground);

	bool** t = new bool*[num];

	for (int k = 0; k < camera.size(); k++)
	{
		for (int j = 0; j < camera[k]->Tnum; j++)
		{
			if (camera[k]->T.at<int>(j, 0) == 0)
			{
				for (int i = j; i < camera[k]->Tnum - 1; i++)
				{
					camera[k]->T.at<int>(i, 0) = camera[k]->T.at<int>(i + 1, 0);
					camera[k]->T.at<int>(i, 1) = camera[k]->T.at<int>(i + 1, 1);
					camera[k]->T.at<int>(i, 2) = camera[k]->T.at<int>(i + 1, 2);
				}
				break;
			}
		}
	}

	int totalNum = 0;
	for (int k = 0; k < camera.size(); k++)
	{
		camera[k]->Tnum--;
		totalNum += camera[k]->Tnum;
	}

	for (int i = 0; i < num; i++)
		t[i] = new bool[totalNum];

	int idx;
	for (int i = 0; i < num; i++)
	{
		idx = 0;
		for (int k = 0; k < camera.size(); k++)
			for (int j = 0; j < camera[k]->Tnum; j++, idx++)
				t[i][idx] = camera[k]->T.at<int>(j, 0) >> i & 0x01;
	}
	showtable(totalNum, num, state, t, camera);

	cout << "***********delete small sunregion*********" << endl << endl;
	idx = 0;
	for (int k = 0; k < camera.size(); k++)
	{
		for (int i = 0; i < camera[k]->Tnum; i++, idx++)
			if ((double)camera[k]->T.at<int>(i, 2) / camera[k]->T.at<int>(i, 1) < 0.20 || (double)camera[k]->T.at<int>(i, 2) < 150)
				for (int j = 0; j < num; j++)
					t[j][idx] = 0;
	}

	for (int j = 0; j < totalNum; j++)
	{
		countcolumn[j] = 0;
		for (int i = 0; i < num; i++)
		{
			if (t[i][j] == 1)
			{
				countcolumn[j] = 1;
				continue;
			}
		}
	}

	shorttable = 1;

	showtable(totalNum, num, state, t, camera);

	//start petrick's method
	vector<string> v;
	string tempstring;

	for (int i = 0; i < totalNum; i++)
	{
		for (int j = 0; j < num; j++)
		{
			if (t[j][i] == 1)
				tempstring = tempstring + (char)(j);
		}
		if (tempstring.size() != 0)
			v.push_back(tempstring);
		tempstring.clear();
	}

	if (v.size() > 0)
	{
		petrick a(v, prob);
		v = a.run();

		double maxprob = 0;;
		double tempprob = 1;
		int idx;

		if (v.size() == 1)
		{
			for (int i = 0; i < v[0].size(); i++)
			{
				state[(int)v[0][i]] = 1;
			}
		}
		else
		{
			for (int k = 0; k < v.size(); k++)
			{
				tempprob = 1;
				for (int i = 0; i < v[k].size(); i++)
				{
					tempprob *= prob[(int)v[k][i]];
				}
				if (tempprob > maxprob)
				{
					maxprob = tempprob;
					idx = k;
				}
			}

			for (int i = 0; i < v[idx].size(); i++)
			{
				state[(int)v[idx][i]] = 1;
			}
		}

	}

	for (int i = 0; i < num; i++)
		delete[] t[i];
	delete[] t;

	for (int i = 0; i < num; i++)
	{
		itoa(i, text, 10);

		if (state[i] == 1)
			cout << i << " ";
		if (i < 16)
		{
			//show candidate number in each view
			for (int k = 0; k < camera.size(); k++)
			{
				cvPutText(&IplImage(camera[k]->frame), text, cvPoint(10 + 20 * i, 250), &font, colorful[i]);
			}



			if (state[i] != 1)
			{
				//for (int k = 0; k < camera.size(); k++)
				//{
				//	CvBlob rectb;




				//	if (st->at<uchar>(0, i) >> k & 0x01)
				//	{
				//		//rectangle(camera1.frame,mea[i].R1, colorful[i]);
				//		rectb.x = camera[k]->r[i].x + camera[k]->r[i].width / 2;
				//		rectb.y = camera[k]->r[i].y + camera[k]->r[i].height / 2;
				//		rectb.w = camera[k]->r[i].width;
				//		rectb.h = camera[k]->r[i].height;

				//		drawDashRect(&IplImage(camera[k]->frame), 1, 4, &rectb, colorful[i], 2);
				//	}
				//}
			}



			else
			{
				//CvBlob rectb;
				result.push_back(i);
				//for (int k = 0; k < camera.size(); k++)
				//	if (st->at<uchar>(0, i) >> k & 0x01)
				//	{
						//rectangle(camera[k]->frame, camera[k]->r[i], colorful[i], 2);
						//cvPutText(&IplImage(camera[k]->frame), text, cvPoint(camera[k]->r[i].x, camera[k]->r[i].y), &font, colorful[i]);
					//}
			}
		}
		else
		{
			//for (int k = 0; k < camera.size(); k++)
				//cvPutText(&IplImage(camera[k]->frame), text, cvPoint(20 * (i - 20), 270), &font, colorful[i]);

			if (state[i] != 1)
			{
				for (int k = 0; k < camera.size(); k++)
				{
					CvBlob rectb;
					if (st->at<uchar>(0, i) >> k & 0x01)
					{
						//rectangle(camera1.frame,mea[i].R1, colorful[i]);
						rectb.x = camera[k]->r[i].x + camera[k]->r[i].width / 2;
						rectb.y = camera[k]->r[i].y + camera[k]->r[i].height / 2;
						rectb.w = camera[k]->r[i].width;
						rectb.h = camera[k]->r[i].height;

						drawDashRect(&IplImage(camera[k]->frame), 1, 4, &rectb, colorful[i], 2);
					}
				}
			}
			else
			{
				result.push_back(i);
				//for (int k = 0; k < camera.size(); k++)
				//	if (st->at<uchar>(0, i) >> k & 0x01)
				//	{
						//rectangle(camera[k]->frame, camera[k]->r[i], colorful[i], 2);
						//cvPutText(&IplImage(camera[k]->frame), text, cvPoint(camera[k]->r[i].x, camera[k]->r[i].y), &font, colorful[i]);
					//}
			}
		}
	}
	cout << endl;
}

void QMmap(CvRect rect, Mat* pFrame, int i)
{
	int js = rect.y;
	int ks = rect.x;
	int je = rect.y + rect.height;
	int ke = rect.x + rect.width;
	int add = pow(2, i);

	for (int k = ks; k < ke; k++)
	{
		for (int j = js; j < je; j++)
		{
			pFrame->at<int>(j, k) = pFrame->at<int>(j, k) + add;
		}
	}
}


int RemoveDuplates(Mat* A, Mat*B, Mat* fmask)
{
	int k = 0;
	int lastNumber = 0;
	int mark = 0;
	int count = 0;
	int kk;

	for (int i = 0; i < A->rows; i++)
	{
		for (int j = 0; j < A->cols; j++)
		{
			if (lastNumber != A->at<int>(i, j))
			{
				mark = 0;
				for (int n = 0; n <= k; n++)
				{
					if (B->at<int>(n, 0) == lastNumber)
					{
						B->at<int>(n, 1) = B->at<int>(n, 1) + count;
						count = 0;
						mark = 1;
						break;
					}
				}
				if (mark == 0)
				{
					k++;
					B->at<int>(k, 0) = lastNumber;
					B->at<int>(k, 1) = B->at<int>(k, 1) + count;
					count = 0;
				}
			}
			lastNumber = A->at<int>(i, j);
			count++;
		}
	}

	for (int n = 0; n <= k; n++)
		if (B->at<int>(n, 0) == lastNumber)
			B->at<int>(n, 1) = B->at<int>(n, 1) + count;

	kk = k;
	lastNumber = 0;
	mark = 0;
	count = 0;
	for (int i = 0; i < A->rows; i++)
	{
		for (int j = 0; j < A->cols; j++)
		{
			if (fmask->at<uchar>(i, j) > 0)
			{
				if (lastNumber != A->at<int>(i, j))
				{
					for (int n = 0; n <= k; n++)
					{
						if (B->at<int>(n, 0) == lastNumber)
						{
							B->at<int>(n, 2) = B->at<int>(n, 2) + count;
							count = 0;
							break;
						}
					}
				}
				lastNumber = A->at<int>(i, j);
				count++;
			}
		}
	}

	for (int n = 0; n <= k; n++)
		if (B->at<int>(n, 0) == lastNumber)
			B->at<int>(n, 2) = B->at<int>(n, 2) + count;
	return kk + 1;
}

int removeRow(bool a[], bool b[], int num)
{
	int zeroNum = 0;

	int oneNum = 0;
	int aOneNum = 0;

	for (int i = 0; i < num; i++)
	{
		switch (a[i])
		{
		case 1:
			aOneNum++;
			switch (b[i])
			{
			case 0:
				return 0;
				break;
			case 1:
				oneNum++;
				break;
			default:
				break;
			}
			break;
		case 0:
			zeroNum++;
			switch (b[i])
			{
			case 1:
				oneNum++;
				break;
			}
			break;
		}
	}
	if (zeroNum == num)
		return 0;
	else if (oneNum == aOneNum)
		return 2;
	else
		return 1;

}

void showtable(int totalNum, int num, bool state[], bool** t, vector<Camera*> camera)
{
	if (shorttable == 0)
	{
		cout << endl;
		cout << "\t\b\b\b";
		for (int k = 0; k < camera.size(); k++)
		{
			for (int j = 0; j < camera[k]->Tnum; j++)
				cout << k;
		}
		cout << " ";
		cout << endl;


		for (int i = 0; i < num; i++)
		{
			cout << "I" << i << ":";
			if (state[i] == 1)
				cout << "o";
			//if (i < 10)
			//	cout << "\t\t";
			//else
			cout << "\t\b\b\b";

			for (int j = 0; j < totalNum; j++)
				if (t[i][j] == 0)
					cout << "+";
				else
					cout << "X";
			cout << endl;
		}
		cout << endl;
	}
	else
	{
		cout << endl;
		cout << "\t\b\b\b";
		int n = 0;
		for (int k = 0; k < camera.size(); k++)
		{
			for (int j = 0; j < camera[k]->Tnum; j++)
			{
				if (countcolumn[n] == 1)
					cout << k;
				n++;
			}
		}
		cout << " ";
		cout << endl;


		for (int i = 0; i < num; i++)
		{
			cout << "I" << i << ":";
			if (state[i] == 1)
				cout << "o";
			cout << "\t\b\b\b";

			for (int j = 0; j < totalNum; j++)
			{
				if (countcolumn[j] == 1)
				{
					if (t[i][j] == 0)
						cout << "+";
					else
						cout << "X";
				}
			}
			cout << endl;
		}
		cout << endl;
	}
}

void drawDashRect(CvArr* img, int linelength, int dashlength, CvBlob* blob, CvScalar color, int thickness)
{
	int w = cvRound(blob->w);
	int h = cvRound(blob->h);

	int tl_x = cvRound(blob->x - blob->w / 2);
	int tl_y = cvRound(blob->y - blob->h / 2);

	int totallength = dashlength + linelength;
	int nCountX = w / totallength;
	int nCountY = h / totallength;

	CvPoint start, end;

	start.y = tl_y;
	start.x = tl_x;

	end.x = tl_x;
	end.y = tl_y;

	for (int i = 0; i < nCountX; i++)
	{
		end.x = tl_x + (i + 1)*totallength - dashlength;
		end.y = tl_y;
		start.x = tl_x + i * totallength;
		start.y = tl_y;
		cvLine(img, start, end, color, thickness);
	}
	for (int i = 0; i < nCountX; i++)
	{
		start.x = tl_x + i * totallength;
		start.y = tl_y + h;
		end.x = tl_x + (i + 1)*totallength - dashlength;
		end.y = tl_y + h;
		cvLine(img, start, end, color, thickness);
	}

	for (int i = 0; i < nCountY; i++)
	{
		start.x = tl_x;
		start.y = tl_y + i * totallength;
		end.y = tl_y + (i + 1)*totallength - dashlength;
		end.x = tl_x;
		cvLine(img, start, end, color, thickness);
	}

	for (int i = 0; i < nCountY; i++)
	{
		start.x = tl_x + w;
		start.y = tl_y + i * totallength;
		end.y = tl_y + (i + 1)*totallength - dashlength;
		end.x = tl_x + w;
		cvLine(img, start, end, color, thickness);
	}
	start.x = tl_x + w;
	start.y = tl_y + h;
	end.x = tl_x + w;
	end.y = tl_y + h - 2;
	cvLine(img, start, end, color, thickness);

	end.x = tl_x + w - 2;
	end.y = tl_y + h;
	cvLine(img, start, end, color, thickness);
}


float bbOverlap(Rect box1, Rect box2)
{
	if (box1.x > box2.x + box2.width) { return 0.0; }
	if (box1.y > box2.y + box2.height) { return 0.0; }
	if (box1.x + box1.width < box2.x) { return 0.0; }
	if (box1.y + box1.height < box2.y) { return 0.0; }
	float colInt = min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x);
	float rowInt = min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y);
	float intersection = colInt * rowInt;
	float area1 = box1.width*box1.height;
	float area2 = box2.width*box2.height;
	return intersection / (area1 + area2 - intersection);
}

bool IsRectCross(const Point &p1, const Point &p2, const Point &q1, const Point &q2)
{
	bool ret = min(p1.x, p2.x) <= max(q1.x, q2.x) &&
		min(q1.x, q2.x) <= max(p1.x, p2.x) &&
		min(p1.y, p2.y) <= max(q1.y, q2.y) &&
		min(q1.y, q2.y) <= max(p1.y, p2.y);
	return ret;
}

Point GetFoot(
	const Point &pt,
	const Point &begin,
	const Point &end)
{
	Point retVal;

	if ((pt.x >= min(begin.x, end.x) && pt.x <= max(begin.x, end.x)) || (pt.y >= min(begin.y, end.y) && pt.y <= max(begin.y, end.y)))
	{
		if ((begin.x - end.x) == 0)
		{
			if (pt.x == begin.x)
			{
				retVal = Point(-1, -1);
				return retVal;
			}
		}

		double dx = begin.x - end.x;
		double dy = begin.y - end.y;
		if (abs(dx) < 0.00000001 && abs(dy) < 0.00000001)
		{
			retVal = begin;
			return retVal;
		}

		double u = (pt.x - begin.x)*(begin.x - end.x) +
			(pt.y - begin.y)*(begin.y - end.y);
		u = u / ((dx*dx) + (dy*dy));

		retVal.x = begin.x + u * dx;
		retVal.y = begin.y + u * dy;

		if (retVal.x > max(begin.x, end.x) || retVal.x < min(begin.x, end.x) || retVal.y > max(begin.y, end.y) || retVal.y < min(begin.y, end.y))
		{
			return Point(-1, -1);
		}
		else
		{
			return retVal;
		}
	}
	else
	{
		return Point(-1, -1);
	}

}

int dfs(int i)
{
	for (int j = 0; j < nd; j++)
	{
		if (!visit[j] && match[i][j])
		{
			visit[j] = 1;
			if (mark[j] == -1 || dfs(mark[j]))
			{
				mark[j] = i;
				return 1;
			}
		}
	}
	return 0;
}

int Hungary()
{
	int ans = 0;
	memset(mark, -1, sizeof(mark));
	for (int i = 0; i < ng; i++)
	{
		memset(visit, 0, sizeof(visit));
		if (dfs(i))
		{
			ans++;
		}
	}
	return ans;
}

Point findLineCross2(vector < vector<int> >  lines)
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

double determinant(double v1, double v2, double v3, double v4)
{
	return (v1*v4 - v2 * v3);
}

bool intersect(Point aa, Point bb, Point cc, Point dd)
{
	double delta = determinant(bb.x - aa.x, cc.x - dd.x, bb.y - aa.y, cc.y - dd.y);
	if (delta <= (1e-6) && delta >= -(1e-6))
	{
		return false;
	}
	double namenda = determinant(cc.x - aa.x, cc.x - dd.x, cc.y - aa.y, cc.y - dd.y) / delta;
	if (namenda > 1 || namenda < 0)
	{
		return false;
	}
	double miu = determinant(bb.x - aa.x, cc.x - aa.x, bb.y - aa.y, cc.y - aa.y) / delta;
	if (miu > 1 || miu < 0)
	{
		return false;
	}
	return true;
}

void Nrec(Mat frm, vector<Camera*> camera, Point xcb, Point vr, int k, const cv::Scalar &color)
{
	double wox, woy, ibx, iby, itx, ity;

	wox = (xcb.x - 250) * 30;
	woy = (xcb.y - 250) * 30;
	camera[k]->cam->worldToImage(wox, woy, 0, ibx, iby);
	ibx = ibx / 2;
	iby = iby / 2;
	camera[k]->cam->worldToImage(wox, woy, PHeight, itx, ity);
	itx = itx / 2;
	ity = ity / 2;

	double widr, heihr;
	heihr = iby - ity;
	widr = heihr * 0.35;

	if (!(ibx - widr / 2 > camera[k]->frame.cols || ibx + widr / 2 < 0 || iby < 0 || ity > camera[k]->frame.rows))
	{
		rectangle(frm, Rect(Point(ibx - widr / 2, ity), Point(ibx + widr / 2, iby)), color, 2);
	}
}