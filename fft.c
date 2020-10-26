#include "fft.h"
#define M_PI_D   (3.141592653589793238462643383279502884197169399375)

void audio_fft_complex(float *data, int N)
{
	int l = (int)ceil(log2(N));
	RDFTContext *context = av_rdft_init(l, DFT_R2C);
	av_rdft_calc(context, data);
	av_rdft_end(context);
}

/* Must be alphabetically ordered for binary search */
const char *fft_window_strings[] = {
	"bartlett",
	"blackmann",
	"blackmann_exact",
	"blackmann_harris",
	"blackmann_nuttall",
	"flat_top",
	"hann",
	"nuttall",
	"sine",
	"triangular",
	"welch"
};

enum fft_windowing_type get_window_type(const char *window)
{
	enum fft_windowing_type ret;
	if (window) {
		int low_bound = 0;
		int high_bound = end_fft_enum - 1;
		int i;
		int c;
		for (; low_bound <= high_bound;) {
			i = (low_bound + ((high_bound - low_bound) / 2));
			c = strcmp(window, fft_window_strings[i]);
			if (c == 0) {
				return i;
			} else if (c > 0) {
				low_bound = i + 1;
			} else {
				high_bound = i - 1;
			}
		}
		return none;
	} else {
		ret = none;
	}
	return ret;
}

/* from: https://en.wikipedia.org/wiki/Window_function */
void window_function(float *data, int N, enum fft_windowing_type type)
{
	int n;
	double d;
	size_t N2;
	double a;
	double a0;
	double a1;
	double a2;
	double a3;
	double a4;
	switch (type) {
	case triangular:
		N2 = N - 1;
		for (n = 0; n < N; n++) {
			d = 1.0 - fabs((n - (N2) / 2.0) / (N / 2.0));
			data[n] *= (float)d;
		}
		break;
	case bartlett:
		N2 = N - 1;
		for (n = 0; n < N; n++) {
			d = 1 - fabs((n - (N2) / 2.0) / (N2 / 2.0));
			data[n] *= (float)d;
		}
		break;
	case welch:
		N2 = N - 1;
		for (n = 0; n < N; n++) {
			d = 1 - pow((n - N2 / 2.0) / (N2 / 2.0), 2);
			data[n] *= (float)d;
		}
	case hann:
		N2 = N - 1;
		a0 = 0.5;
		a1 = 0.5;
		for (n = 0; n < N; n++) {
			d = a0 - a1 * cos((2.0 * M_PI_D * n) / N2);
			data[n] *= (float)d;
		}
		break;
	case blackmann:
		a = 0.16;
		a0 = (1 - a) / 2.0;
		a1 = 0.5;
		a2 = a / 2.0;
		goto cossum2;
	case blackmann_exact:
		a0 = 7938.0 / 18608.0;
		a1 = 9240.0 / 18608.0;
		a2 = 1430.0 / 18608.0;
		goto cossum2;
	case nuttall:
		a0 = 0.355768;
		a1 = 0.487396;
		a2 = 0.144232;
		a3 = 0.012604;
		goto cossum3;
	case blackmann_nuttall:
		a0 = 0.3635819;
		a1 = 0.4891775;
		a2 = 0.1365995;
		a3 = 0.0106411;
		goto cossum3;
	case blackmann_harris:
		a0 = 0.35875;
		a1 = 0.48829;
		a2 = 0.14128;
		a3 = 0.01168;
		goto cossum3;
	case flat_top:
		a0 = 1;
		a1 = 1.93;
		a2 = 1.29;
		a3 = 0.388;
		a4 = 0.028;
		goto cossum4;
	default:
		return;
	}
	return;

cossum2:
	N2 = N - 1;
	for (n = 0; n < N; n++) {
		d = a0 - a1 * cos((2.0 * M_PI_D * n) / N2) +
			a2 * cos((4.0 * M_PI_D * n) / N2);
		data[n] *= (float)d;
	}
	return;
cossum3:
	N2 = N - 1;
	for (n = 0; n < N; n++) {
		d = a0 - a1 * cos((2.0 * M_PI_D * n) / N2) +
			a2 * cos((4.0 * M_PI_D * n) / N2) -
			a3 * cos((6.0 * M_PI_D * n) / N2);
		data[n] *= (float)d;
	}
	return;
cossum4:
	N2 = N - 1;
	for (n = 0; n < N; n++) {
		d = a0 - a1 * cos((2.0 * M_PI_D * n) / N2) +
			a2 * cos((4.0 * M_PI_D * n) / N2) -
			a3 * cos((6.0 * M_PI_D * n) / N2) +
			a4 * cos((8.0 * M_PI_D * n) / N2);
		data[n] *= (float)d;
	}
	return;
}
