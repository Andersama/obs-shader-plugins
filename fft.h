#pragma once

#include <obs.h>
#include <math.h>
#include <libavcodec/avfft.h>

#ifdef __cplusplus
extern "C" {
#endif

/*Should be alphabetically ordered*/
enum fft_windowing_type {
	none = -1,
	rectangular = -1,
	bartlett,
	blackmann,
	blackmann_exact,
	blackmann_harris,
	blackmann_nuttall,
	flat_top,
	hann,
	nuttall,
	sine,
	triangular,
	welch,
	end_fft_enum
};

void audio_fft_complex(float* X, int N);
enum fft_windowing_type get_window_type(const char *window);
void window_function(float *data, int N, enum fft_windowing_type type);

#ifdef __cplusplus
}
#endif
