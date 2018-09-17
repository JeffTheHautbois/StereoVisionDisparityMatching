#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv/cv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <chrono>



int main(int argc, char **argv) {


	std::string pianoL = "Piano-perfect-im0-s.png";
	std::string pianoR = "Piano-perfect-im1-s.png";
	std::string motorcycleL = "Motorcycle-perfect-im0-s.png";
	std::string motorcycleR = "Motorcycle-perfect-im1-s.png";
	std::string umbrellaL = "Umbrella-perfect-im0-s.png";
	std::string umbrellaR = "Umbrella-perfect-im1-s.png";

	//flag variables for mode control
	int imgFlag = 0;
	int approachFlag = 2;
	int loadGrayScale = 0;

	//threshHold presets
	int pianoMaxDisp = 70;
	int motorcycleMaxDisp = 60;
	int umbrellaMaxDisp = 65;
	int minDisp = 0;
	int maxDisp = 70; //default

	//load img assuming both image are the same resolution
	cv::Mat imgL, imgR;
	if (loadGrayScale){
		if (imgFlag == 0){
			imgL = cv::imread(pianoL, cv::IMREAD_GRAYSCALE);
			imgR = cv::imread(pianoR, cv::IMREAD_GRAYSCALE);
			//set min/maxDisparity to search
			maxDisp = pianoMaxDisp;
		} else if (imgFlag == 1){
			imgL = cv::imread(motorcycleL, cv::IMREAD_GRAYSCALE);
			imgR = cv::imread(motorcycleR, cv::IMREAD_GRAYSCALE);
			maxDisp = motorcycleMaxDisp;
		} else if (imgFlag == 2){
			imgL = cv::imread(umbrellaL, cv::IMREAD_GRAYSCALE);
			imgR = cv::imread(umbrellaR, cv::IMREAD_GRAYSCALE);
			maxDisp = umbrellaMaxDisp;
		}
	} else {
		if (imgFlag == 0){
			imgL = cv::imread(pianoL, CV_8UC1);
			imgR = cv::imread(pianoR, CV_8UC1);
			//set min/maxDisparity to search
			maxDisp = pianoMaxDisp;
		} else if (imgFlag == 1){
			imgL = cv::imread(motorcycleL, CV_8UC1);
			imgR = cv::imread(motorcycleR, CV_8UC1);
			maxDisp = motorcycleMaxDisp;
		} else if (imgFlag == 2){
			imgL = cv::imread(umbrellaL, CV_8UC1);
			imgR = cv::imread(umbrellaR, CV_8UC1);
			maxDisp = umbrellaMaxDisp;
		}
	}

	//cv::namedWindow("imgL", CV_WINDOW_AUTOSIZE);
	//cv::imshow("imgL", imgL);

	int imgRows = imgL.rows;
	int imgCols = imgL.cols;

	//create empty disparity matrix
	cv::Mat disp = cv::Mat::zeros(imgRows,imgCols,CV_8UC1);

	//set maskSize
	int maskSize = 7;
	int maskOffset = (maskSize - 1) / 2;

//	//createPaddedImg
//	cv::Mat imgL_pad, imgR_pad;
//	cv::copyMakeBorder(imgL,imgL_pad,maskOffset,maskOffset,maskOffset,maskOffset,cv::BORDER_CONSTANT,cv::Scalar(0));
//	cv::copyMakeBorder(imgR,imgR_pad,maskOffset,maskOffset,maskOffset,maskOffset,cv::BORDER_CONSTANT,cv::Scalar(0));


	int minSSD = 99999;
	int tempSSD = 0;
	int dist = 0;
	int pixelDiff = 0;
	int bestMatch = 0;

	int endRowIndex = imgRows - maskOffset;
	int endColIndex = imgCols - maskOffset;

	//start clocking
	auto start = std::chrono::high_resolution_clock::now();

	//first approach (long run time)
	if (approachFlag == 1){
		for (int i = maskOffset; i < endRowIndex; i++){
			std::cout<< "." << "1\n";
			for (int j = maskOffset; j < endColIndex; j++){
				minSSD = 99999;
				dist = 0;
				for (int k = maskOffset; k < endColIndex; k++){
					tempSSD = 0;
					pixelDiff = 0;
					for (int m = -maskOffset; m < maskOffset; m++){
						for (int n = -maskOffset; n < maskOffset; n++){
							pixelDiff = (int)imgL.at<uchar>(i+m,j+n) - (int)imgR.at<uchar>(i+m,k+n);
							tempSSD += (pixelDiff * pixelDiff);
						}
					}
					if (minSSD > tempSSD){
						minSSD = tempSSD;
						dist = abs(k-j);
					}
				}
				if (dist > 255){
					disp.at<uchar>(i,j) = 0;
				} else {
					disp.at<uchar>(i,j) = dist;
				}
			}
		}
	//second approach
	} else if (approachFlag == 2){
		int checkVal = 0;
		for (int i = maskOffset; i < endRowIndex; i++){
			std::cout<< "." << "2\n";
			for (int j = maskOffset; j < endColIndex; j++){
				//calculate if col index will be out of bound
				minSSD = 99999;
				checkVal = j + maxDisp;
				bestMatch = j;
				if (checkVal < endColIndex){
					//index is within bound
					for (int k = minDisp; k < maxDisp; k++){
						tempSSD = 0;
						pixelDiff = 0;
						for (int m = -maskOffset; m < maskOffset; m++){
							for (int n = -maskOffset; n < maskOffset; n++){
								pixelDiff = (int)imgL.at<uchar>(i+m,j+n) - (int)imgR.at<uchar>(i+m,j+n-k);
								tempSSD += (pixelDiff * pixelDiff);
							}
						}
						if (minSSD > tempSSD){
							minSSD = tempSSD;
							bestMatch = k;
						}
					}
				} else {
					//index is out of bound
					for (int k = minDisp; k < maxDisp - (checkVal-endColIndex) - 1; k++){
						tempSSD = 0;
						pixelDiff = 0;
						for (int m = -maskOffset; m < maskOffset; m++){
							for (int n = -maskOffset; n < maskOffset; n++){
								pixelDiff = (int)imgL.at<uchar>(i+m,j+n) - (int)imgR.at<uchar>(i+m,j+n-k);
								tempSSD += (pixelDiff * pixelDiff);
							}
						}
						if (minSSD > tempSSD){
							minSSD = tempSSD;
							bestMatch = k;
						}
					}
				}
				disp.at<uchar>(i,j) = bestMatch;
			}
		}
	}


	cv::equalizeHist(disp,disp);
	cv::normalize(disp, disp,255, 0, cv::NORM_MINMAX);
	cv::namedWindow("disp", CV_WINDOW_AUTOSIZE);
	cv::imshow("disp", disp);

	//finish clock
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time: " << elapsed.count() << " s\n";
	std::cout << "finished" << std::endl;

	//convert to point cloud

	//since all images are aligned fx = fy
	double f, cx0, cx1, cy, baseline, doffs;
	double x,y,z;

	std::ifstream infile;
	if (imgFlag == 0){
		infile.open("pianoCalib.txt");
	} else if (imgFlag == 1){
		infile.open("motorcycleCalib.txt");
	} else if (imgFlag == 2){
		infile.open("umbrellaCalib.txt");
	}

	std::string word;
	std::string temp;

	infile >> word;
	temp = word.substr(6);

	//focal point value
	f = std::stod(temp);
	f = f/4.0;
	std::cout << f << std::endl;
	infile >> word;
	infile >> word;

	//cam0 center x value
	cx0 = std::stod(word);
	cx0 = cx0/4.0;
	std::cout << cx0 << std::endl;
	infile >> word;
	infile >> word;
	infile >> word;

	//center y value
	cy = std::stod(word);
	cy = cy/4.0;
	std::cout << cy << std::endl;

	infile >> word;
	infile >> word;
	infile >> word;
	infile >> word;
	infile >> word;
	infile >> word;

	//cam1 center x value
	cx1 = std::stod(word);
	cx1 = cx1/4.0;
	std::cout << cx1 << std::endl;

	infile >> word;
	infile >> word;
	infile >> word;
	infile >> word;
	infile >> word;
	infile >> word;
	infile >> word;
	infile >> word;

	temp = word.substr(9);
	baseline = std::stod(temp);
	baseline = baseline/4.0;
	std::cout << baseline << std::endl;

	infile.close();
	std::ofstream outfile;
	if (imgFlag == 0){
		outfile.open("piano3D.txt");
	} else if (imgFlag == 1){
		outfile.open("motorcycle3D.txt");
	} else if (imgFlag == 2){
		outfile.open("umbrella3D.txt");
	}

	for (int u = 0; u < imgCols; u++ ){
		for (int v = 0; v < imgRows; v++){
			z = baseline * f / ((double)disp.at<uchar>(v,u)+(cx1-cx0));
			x = u - cx0;
			y = v - cy;
			outfile << x << ',' << y << ',' << z << "\n";
		}
	}
	outfile.close();


	cv::waitKey();
	return 0;
}
