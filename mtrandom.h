#pragma once

/*
double random_double_cpp(double min, double max);
int random_int_cpp(int min, int max);
*/

#ifdef __cplusplus
extern "C"
{
#endif
	
double random_double(double min, double max);
int random_int(int min, int max);

#ifdef __cplusplus
}
#endif