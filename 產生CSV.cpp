#include<iostream>
#include<windows.h>
#include <string> 
#include<fstream>
#include<sstream>

using namespace std;

int main()
{
	ofstream file;
	//writefile
	file.open("wally_all.csv");
	for (int i = 1; i <= 100; i++)
	{
		file << "./number_data/train_img/0(" << i << ").jpg" << ',' << "1" << endl;
	}
	for (int i = 1; i <= 100; i++)
	{
		file << "./number_data/train_img/1(" << i << ").jpg" << ',' << "1" << endl;
	}
	for (int i = 1; i <= 100; i++)
	{
		file << "./number_data/train_img/2(" << i << ").jpg" << ',' << "1" << endl;
	}
	for (int i = 1; i <= 100; i++)
	{
		file << "./number_data/train_img/3(" << i << ").jpg" << ',' << "1" << endl;
	}
	for (int i = 1; i <= 100; i++)
	{
		file << "./number_data/train_img/4(" << i << ").jpg" << ',' << "1" << endl;
	}
	for (int i = 1; i <= 100; i++)
	{
		file << "./number_data/train_img/5(" << i << ").jpg" << ',' << "1" << endl;
	}
	for (int i = 1; i <= 100; i++)
	{
		file << "./number_data/train_img/6(" << i << ").jpg" << ',' << "1" << endl;
	}
	for (int i = 1; i <= 100; i++)
	{
		file << "./number_data/train_img/7(" << i << ").jpg" << ',' << "1" << endl;
	}
	for (int i = 1; i <= 100; i++)
	{
		file << "./number_data/train_img/8(" << i << ").jpg" << ',' << "1" << endl;

	}
	for (int i = 1; i <= 100; i++)
	{
		file << "./number_data/train_img/9(" << i << ").jpg" << ',' << "1" << endl;
	}
	file.close();
	return 0;
}
