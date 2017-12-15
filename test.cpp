#include "Panoramic_Stitching.h"
void run_data(std::string form);

void main()
{
	std::string type;
	//std::cout << "input your image type: ";////jpg,png
	//std::cin >> type;
	type= "*.jpg";
	run_data(type);
}

void run_data(std::string form)
{
	Panor::CPanoramic PS("data/", form);
	PS.ReadImg();
	//cv::Mat HH=PS.image_transform(1,2);
	PS.test();

}