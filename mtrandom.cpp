#include "mtrandom.h"

#include <random>
using namespace std;

random_device rd{};
mt19937 engine{ rd() };

double random_double(double min, double max)
{
	std::uniform_real_distribution<double> dist(min, max);
	return dist(engine);	
}

int random_int(int min, int max)
{
	std::uniform_int_distribution<int> dist(min, max);
	return dist(engine);
}