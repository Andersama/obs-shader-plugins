#include <graphics/graphics.h>
#include <graphics/image-file.h>
#include <obs-module.h>
#include <util/base.h>
#include <util/circlebuf.h>
#include <util/dstr.h>
#include <util/platform.h>
#include <util/threading.h>

#include <float.h>
#include <limits.h>
#include <stdio.h>

#include "fft.h"
#include "tinyexpr.h"
#include "mtrandom.h"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs_shader_filter", "en-US")
#define blog(level, msg, ...) blog(level, "shader-filter: " msg, ##__VA_ARGS__)

#define _MT obs_module_text

static void sidechain_capture(void *p, obs_source_t *source,
		const struct audio_data *audio_data, bool muted);

bool is_power_of_two(size_t val)
{
	return val != 0 && (val & (val - 1)) == 0;
}

double hlsl_clamp(double in, double min, double max)
{
	if (in < min)
		return min;
	if (in > max)
		return max;
	return in;
}

#define M_PI_D 3.141592653589793238462643383279502884197169399375
double hlsl_degrees(double radians)
{
	return radians * (180.0 / M_PI_D);
}

double hlsl_rad(double degrees)
{
	return degrees * (M_PI_D / 180.0);
}

double audio_mel_from_hz(double hz)
{
	return 2595 * log10(1 + hz / 700.0);
}

double audio_hz_from_mel(double mel)
{
	return 700 * (pow(10, mel / 2595) - 1);
}

const static double flt_max = FLT_MAX;
const static double flt_min = FLT_MIN;
const static double int_min = INT_MIN;
const static double int_max = INT_MAX;
static double sample_rate;
static double output_channels;

/* Additional likely to be used functions for mathmatical expressions */
void prep_te_funcs(struct darray *te_vars)
{
	te_variable funcs[] = {{"clamp", hlsl_clamp, TE_FUNCTION3},
			{"float_max", &flt_max}, {"float_min", &flt_min},
			{"int_max", &int_max}, {"int_min", &int_min},
			{"sample_rate", &sample_rate},
			{"channels", &output_channels},
			{"mel_from_hz", audio_mel_from_hz, TE_FUNCTION1},
			{"hz_from_mel", audio_hz_from_mel, TE_FUNCTION1},
			{"degrees", hlsl_degrees, TE_FUNCTION1},
			{"radians", hlsl_rad, TE_FUNCTION1},
			{"random", random_double, TE_FUNCTION2} };
	darray_push_back_array(sizeof(te_variable), te_vars, &funcs[0], 12);
}

void append_te_variable(struct darray *te_vars, te_variable *v)
{
	darray_push_back(sizeof(te_variable), te_vars, v);
}

void clear_te_variables(struct darray *te_vars)
{
	darray_free(te_vars);
}

/* Gets the integer value of an annotation (presuming its type is numeric) */
int get_annotation_int(gs_eparam_t *annotation, int default_value)
{
	struct gs_effect_param_info note_info;
	gs_effect_get_param_info(annotation, &note_info);

	void *val = NULL;
	int ret   = default_value;

	if (annotation) {
		if (note_info.type == GS_SHADER_PARAM_FLOAT) {
			val = (void *)gs_effect_get_default_val(annotation);
			if (val) {
				ret = (int)*((float *)val);
				bfree(val);
				val = NULL;
			}
		} else if (note_info.type == GS_SHADER_PARAM_INT ||
				note_info.type == GS_SHADER_PARAM_BOOL) {
			val = (void *)gs_effect_get_default_val(annotation);
			if (val) {
				ret = *((int *)val);
				bfree(val);
				val = NULL;
			}
		}
	}

	return ret;
}

int get_eparam_int(gs_eparam_t *param, const char *name, int default_value)
{
	gs_eparam_t *note = gs_param_get_annotation_by_name(param, name);
	return get_annotation_int(note, default_value);
}

/* Boolean values are stored as integers (so...we cast) */
bool get_annotation_bool(gs_eparam_t *annotation, bool default_value)
{
	int val = get_annotation_int(annotation, (int)default_value);
	return val != 0;
}

/* Gets an annotion by name and returns its boolean value */
bool get_eparam_bool(gs_eparam_t *param, const char *name, bool default_value)
{
	gs_eparam_t *note = gs_param_get_annotation_by_name(param, name);
	return get_annotation_bool(note, default_value);
}

/* Gets the floating point value of an annotation */
float get_annotation_float(gs_eparam_t *annotation, float default_value)
{
	struct gs_effect_param_info note_info;
	gs_effect_get_param_info(annotation, &note_info);

	void *val = NULL;
	float ret = default_value;

	if (annotation) {
		if (note_info.type == GS_SHADER_PARAM_FLOAT) {
			val = (void *)gs_effect_get_default_val(annotation);
			if (val) {
				ret = *((float *)val);
				bfree(val);
				val = NULL;
			}
		} else if (note_info.type == GS_SHADER_PARAM_INT ||
				note_info.type == GS_SHADER_PARAM_BOOL) {
			val = (void *)gs_effect_get_default_val(annotation);
			if (val) {
				ret = (float)*((int *)val);
				bfree(val);
				val = NULL;
			}
		}
	}

	return ret;
}

/* Gets an annotion by name and returns its floating point value */
float get_eparam_float(
		gs_eparam_t *param, const char *name, float default_value)
{
	gs_eparam_t *note = gs_param_get_annotation_by_name(param, name);
	return get_annotation_float(note, default_value);
}

/* Free w/ bfree */
char *get_annotation_string(gs_eparam_t *annotation, const char *default_value)
{
	struct gs_effect_param_info note_info;
	gs_effect_get_param_info(annotation, &note_info);

	char *val = NULL;

	if (annotation && note_info.type == GS_SHADER_PARAM_STRING) {
		val = (char *)gs_effect_get_default_val(annotation);
		if (val)
			return val;
	}

	return bstrdup(default_value);
}

/* Free w/ bfree */
char *get_eparam_string(
		gs_eparam_t *param, const char *name, const char *default_value)
{
	gs_eparam_t *note = gs_param_get_annotation_by_name(param, name);
	return get_annotation_string(note, default_value);
}

/* Struct for dealing w/ integers like the vec4 */
struct long4 {
	union {
		struct {
			int32_t x, y, z, w;
		};
		int32_t ptr[4];
		__m128 m;
	};
};

/* Struct for dealing w/ converting vec2 floating points to doubles */
struct double2 {
	union {
		struct {
			double x, y;
		};
		double f[2];
	};
};

/* Space saving functions (adding properties happens a lot) */
void obs_properties_add_float_prop(obs_properties_t *props, const char *name,
		const char *desc, double min, double max, double step,
		bool is_slider)
{
	if (is_slider)
		obs_properties_add_float_slider(
				props, name, desc, min, max, step);
	else
		obs_properties_add_float(props, name, desc, min, max, step);
}

void obs_properties_add_int_prop(obs_properties_t *props, const char *name,
		const char *desc, int min, int max, int step, bool is_slider)
{
	if (is_slider)
		obs_properties_add_int_slider(
				props, name, desc, min, max, step);
	else
		obs_properties_add_int(props, name, desc, min, max, step);
}

void obs_properties_add_numerical_prop(obs_properties_t *props,
		const char *name, const char *desc, double min, double max,
		double step, bool is_slider, bool is_float)
{
	if (is_float)
		obs_properties_add_float_prop(
				props, name, desc, min, max, step, is_slider);
	else
		obs_properties_add_int_prop(props, name, desc, (int)min,
				(int)max, (int)step, is_slider);
}

void dstr_copy_cat(struct dstr *str, const char *start, const char *mid,
		const char *end, size_t end_len)
{
	dstr_free(str);
	dstr_copy(str, start);
	dstr_cat(str, mid);
	dstr_ncat(str, end, end_len);
}

void obs_properties_add_vec_prop(obs_properties_t *props, const char *name,
		const char *desc, double min, double max, double step,
		bool is_slider, bool is_float, int vec_num)
{
	const char *mixin = "xyzw";
	struct dstr n_param_name;
	struct dstr n_param_desc;
	dstr_init(&n_param_name);
	dstr_init(&n_param_desc);
	int vec_count = vec_num <= 4 ? (vec_num >= 0 ? vec_num : 0) : 4;
	for (int i = 0; i < vec_count; i++) {
		dstr_copy_cat(&n_param_name, name, ".", mixin + i, 1);
		dstr_copy_cat(&n_param_desc, desc, ".", mixin + i, 1);

		obs_properties_add_numerical_prop(props, n_param_name.array,
				n_param_desc.array, min, max, step, is_slider,
				is_float);
	}
	dstr_free(&n_param_name);
	dstr_free(&n_param_desc);
}

void obs_properties_add_vec_array(obs_properties_t *props, const char *name,
		const char *desc, double min, double max, double step,
		bool is_slider, int vec_num)
{
	obs_properties_add_vec_prop(props, name, desc, min, max, step,
			is_slider, true, vec_num);
}

void obs_properties_add_int_array(obs_properties_t *props, const char *name,
		const char *desc, double min, double max, double step,
		bool is_slider, int vec_num)
{
	obs_properties_add_vec_prop(props, name, desc, min, max, step,
			is_slider, false, vec_num);
}

/*
 * functions to extract lists from annotations
 *	< [int|float|bool|string] list_item = ?; string list_item_?_name = "" >
 */
void fill_int_list(obs_property_t *p, gs_eparam_t *param)
{
	uint64_t notes_count = gs_param_get_num_annotations(param);
	struct gs_effect_param_info info;

	bool uses_module_text =
			get_eparam_bool(param, "list_module_text", false);

	char *c_tmp = NULL;
	int value   = 0;

	struct dstr name_variable;
	struct dstr value_string;
	dstr_init(&name_variable);
	dstr_init(&value_string);
	dstr_ensure_capacity(&value_string, 21);
	for (uint64_t i = 0; i < notes_count; i++) {
		gs_eparam_t *note = gs_param_get_annotation_by_idx(param, i);
		gs_effect_get_param_info(note, &info);

		if (info.type == GS_SHADER_PARAM_INT ||
				info.type == GS_SHADER_PARAM_FLOAT ||
				info.type == GS_SHADER_PARAM_BOOL) {

			if (astrcmpi_n(info.name, "list_item", 9) == 0) {
				dstr_free(&name_variable);
				dstr_copy(&name_variable, info.name);
				dstr_cat(&name_variable, "_name");

				value = get_annotation_int(note, 0);

				sprintf(value_string.array, "%d", value);

				c_tmp = get_eparam_string(param,
						name_variable.array,
						value_string.array);

				obs_property_list_add_int(p,
						uses_module_text ? _MT(c_tmp)
								 : c_tmp,
						value);

				bfree(c_tmp);
			}
		}
	}
	dstr_free(&name_variable);
	dstr_free(&value_string);
}

void fill_float_list(obs_property_t *p, gs_eparam_t *param)
{
	uint64_t notes_count = gs_param_get_num_annotations(param);
	struct gs_effect_param_info info;

	bool uses_module_text =
			get_eparam_bool(param, "list_module_text", false);

	char *c_tmp = NULL;
	float value = 0;

	struct dstr name_variable;
	struct dstr value_string;
	dstr_init(&name_variable);
	dstr_init(&value_string);
	dstr_ensure_capacity(&value_string, 21);
	for (uint64_t i = 0; i < notes_count; i++) {
		gs_eparam_t *note = gs_param_get_annotation_by_idx(param, i);
		gs_effect_get_param_info(note, &info);

		if (info.type == GS_SHADER_PARAM_INT ||
				info.type == GS_SHADER_PARAM_FLOAT ||
				info.type == GS_SHADER_PARAM_BOOL) {

			if (astrcmpi_n(info.name, "list_item", 9) == 0) {
				dstr_free(&name_variable);
				dstr_copy(&name_variable, info.name);
				dstr_cat(&name_variable, "_name");

				value = get_annotation_float(note, 0);

				sprintf(value_string.array, "%f", value);

				c_tmp = get_eparam_string(param,
						name_variable.array,
						value_string.array);

				obs_property_list_add_float(p,
						uses_module_text ? _MT(c_tmp)
								 : c_tmp,
						value);

				bfree(c_tmp);
			}
		}
	}
	dstr_free(&name_variable);
	dstr_free(&value_string);
}

void fill_string_list(obs_property_t *p, gs_eparam_t *param)
{
	uint64_t notes_count = gs_param_get_num_annotations(param);
	struct gs_effect_param_info info;

	bool uses_module_text =
			get_eparam_bool(param, "list_module_text", false);

	char *c_tmp = NULL;
	char *value = NULL;

	struct dstr name_variable;
	struct dstr value_string;
	dstr_init(&name_variable);
	dstr_init(&value_string);
	for (uint64_t i = 0; i < notes_count; i++) {
		gs_eparam_t *note = gs_param_get_annotation_by_idx(param, i);
		gs_effect_get_param_info(note, &info);

		if (info.type == GS_SHADER_PARAM_STRING) {

			if (astrcmpi_n(info.name, "list_item_", 10) == 0) {
				dstr_free(&name_variable);
				dstr_copy(&name_variable, info.name);
				if (dstr_find(&name_variable, "_name"))
					continue;
				dstr_cat(&name_variable, "_name");

				value = get_annotation_string(note, "");
				dstr_free(&value_string);
				dstr_copy(&value_string, value);

				c_tmp = get_eparam_string(param,
						name_variable.array,
						value_string.array);

				obs_property_list_add_string(p,
						uses_module_text ? _MT(c_tmp)
								 : c_tmp,
						value);

				bfree(c_tmp);
				bfree(value);
			}
		}
	}
	dstr_free(&name_variable);
	dstr_free(&value_string);
}

/* functions to add sources to a list for use as textures */
bool fill_properties_source_list(void *param, obs_source_t *source)
{
	obs_property_t *p       = (obs_property_t *)param;
	uint32_t flags          = obs_source_get_output_flags(source);
	const char *source_name = obs_source_get_name(source);

	if ((flags & OBS_SOURCE_VIDEO) != 0 && obs_source_active(source))
		obs_property_list_add_string(p, source_name, source_name);

	return true;
}

void fill_source_list(obs_property_t *p)
{
	obs_property_list_add_string(p, _MT("None"), "");
	obs_enum_sources(&fill_properties_source_list, (void *)p);
}

bool fill_properties_audio_source_list(void *param, obs_source_t *source)
{
	obs_property_t *p       = (obs_property_t *)param;
	uint32_t flags          = obs_source_get_output_flags(source);
	const char *source_name = obs_source_get_name(source);

	if ((flags & OBS_SOURCE_AUDIO) != 0 && obs_source_active(source))
		obs_property_list_add_string(p, source_name, source_name);

	return true;
}

void fill_audio_source_list(obs_property_t *p)
{
	obs_property_list_add_string(p, _MT("None"), "");
	obs_enum_sources(&fill_properties_audio_source_list, (void *)p);
}

int obs_get_vec_num(enum gs_shader_param_type type)
{
	switch (type) {
	case GS_SHADER_PARAM_VEC4:
	case GS_SHADER_PARAM_INT4:
		return 4;
	case GS_SHADER_PARAM_VEC3:
	case GS_SHADER_PARAM_INT3:
		return 3;
	case GS_SHADER_PARAM_VEC2:
	case GS_SHADER_PARAM_INT2:
		return 2;
	case GS_SHADER_PARAM_FLOAT:
	case GS_SHADER_PARAM_INT:
		return 1;
	case GS_SHADER_PARAM_MATRIX4X4:
		return 16;
	}
	return 0;
}

struct effect_param_data {
	/* The name and description of an obs property */
	struct dstr name;
	struct dstr desc;
	enum gs_shader_param_type type;
	gs_eparam_t *param;

	gs_image_file_t *image;
	gs_texture_t *texture;

	bool is_vec4;
	bool is_list;
	bool is_source;
	bool is_media;
	bool is_audio_source;
	bool is_fft;
	bool is_psd;

	bool bound;
	bool update_per_frame;

	/* An array of strings for use w/ the array types */
	struct dstr array_names[4];

	obs_source_t *media_source;
	obs_weak_source_t *media_weak_source;
	pthread_mutex_t sidechain_update_mutex;

	gs_texrender_t *texrender;

	bool update_expr_per_frame[4];
	bool has_expr[4];
	struct dstr expr[4];

	/* These store the varieties of values passed to the shader */
	union {
		long long i;
		double f;
		struct vec4 v4;
		struct long4 l4;
	} value;

	/* These hold the above as doubles for use in expressions */
	union {
		double f[4];
	} te_bind;

	size_t max_sidechain_frames;
	size_t max_sidechain_buf_frames;
	pthread_mutex_t sidechain_mutex;
	struct circlebuf sidechain_data[MAX_AUDIO_CHANNELS];
	float *sidechain_buf;
	size_t num_channels;
	size_t fft_bins;
	size_t fft_samples;

	enum fft_windowing_type window;
};

void effect_param_data_release(struct effect_param_data *param)
{
	dstr_free(&param->name);
	dstr_free(&param->desc);

	obs_source_remove_audio_capture_callback(
			param->media_source, sidechain_capture, param);

	obs_source_release(param->media_source);
	param->media_source = NULL;

	obs_enter_graphics();
	gs_texrender_destroy(param->texrender);
	gs_texture_destroy(param->texture);
	gs_image_file_free(param->image);
	obs_leave_graphics();
	param->texrender = NULL;
	param->texture   = NULL;

	bfree(param->image);
	param->image = NULL;

	size_t i;
	for (i = 0; i < 4; i++) {
		dstr_free(&param->array_names[i]);
		dstr_free(&param->expr[i]);
	}

	for (i = 0; i < MAX_AUDIO_CHANNELS; i++)
		circlebuf_free(&param->sidechain_data[i]);

	bfree(param->sidechain_buf);
	param->sidechain_buf = NULL;

	pthread_mutex_destroy(&param->sidechain_mutex);
	pthread_mutex_destroy(&param->sidechain_update_mutex);
}

struct shader_filter_data {
	obs_source_t *context;
	obs_data_t *settings;
	gs_effect_t *effect;

	bool reload_effect;
	struct dstr last_path;

	/* Holds mathmatical expressions to evaluate cropping / expansion */
	union {
		struct {
			struct dstr expr_left, expr_right, expr_top,
					expr_bottom;
		};
		struct dstr expr[4];
	};

	/* These hold variables and functions used to evaluate expressions */
	DARRAY(te_variable) vars;

	gs_eparam_t *param_uv_offset;
	gs_eparam_t *param_uv_scale;
	gs_eparam_t *param_uv_pixel_interval;
	gs_eparam_t *param_elapsed_time;

	union {
		struct {
			int resize_left, resize_right, resize_top,
					resize_bottom;
		};
		struct long4 resize;
	};

	union {
		struct {
			bool bind_left, bind_right, bind_top, bind_bottom;
		};
		bool bind[4];
	};

	bool bind_update_per_frame[4];

	bool show_expansions;

	int total_width;
	int total_height;

	struct vec2 uv_offset;
	struct vec2 uv_scale;
	struct vec2 uv_pixel_interval;

	struct double2 uv_scale_bind;
	struct double2 uv_pixel_interval_bind;

	float elapsed_time;

	union {
		double f;
	} elapsed_time_bind;

	DARRAY(struct effect_param_data) stored_param_list;
	DARRAY(struct effect_param_data*) eval_param_list;
};

void update_filter_cache(struct shader_filter_data *filter, gs_eparam_t *param)
{
	struct gs_effect_param_info info;
	gs_effect_get_param_info(param, &info);

	if (strcmp(info.name, "uv_offset") == 0) {
		filter->param_uv_offset = param;
	} else if (strcmp(info.name, "uv_scale") == 0) {
		filter->param_uv_scale = param;
	} else if (strcmp(info.name, "uv_pixel_interval") == 0) {
		filter->param_uv_pixel_interval = param;
	} else if (strcmp(info.name, "elapsed_time") == 0) {
		filter->param_elapsed_time = param;
	} else if (strcmp(info.name, "ViewProj") == 0) {
		filter->show_expansions = get_eparam_bool(
				param, "show_expansions", false);
	} else if (strcmp(info.name, "image") == 0) {
		/* Nothing. */
	} else {
		struct effect_param_data *cached_data =
				da_push_back_new(filter->stored_param_list);
		dstr_init_copy(&cached_data->name, info.name);
		cached_data->type  = info.type;
		cached_data->param = param;
		if (pthread_mutex_init(&cached_data->sidechain_mutex, NULL) !=
				0) {
			blog(LOG_ERROR, "Failed to create mutex");
			blog(LOG_ERROR, "Removing Param: %s", info.name);
			da_pop_back(filter->stored_param_list);
		}

		if (pthread_mutex_init(&cached_data->sidechain_update_mutex,
				NULL) != 0) {
			pthread_mutex_destroy(&cached_data->sidechain_mutex);
			blog(LOG_ERROR, "Failed to create mutex");
			blog(LOG_ERROR, "Removing Param: %s", info.name);
			da_pop_back(filter->stored_param_list);
		}
	}
}

static void shader_filter_reload_effect(struct shader_filter_data *filter)
{
	/* First, clean up the old effect and all references to it. */
	size_t param_count = filter->stored_param_list.num;
	size_t i;
	for (i = 0; i < param_count; i++) {
		struct effect_param_data *param =
				filter->stored_param_list.array + i;
		effect_param_data_release(param);
	}

	da_free(filter->stored_param_list);
	da_free(filter->eval_param_list);
	/* Clear expression variables, they need to be refreshed */
	da_free(filter->vars);

	filter->param_elapsed_time      = NULL;
	filter->param_uv_offset         = NULL;
	filter->param_uv_pixel_interval = NULL;
	filter->param_uv_scale          = NULL;

	/* Make sure the expressions aren't considered bound yet */
	filter->bind_left   = false;
	filter->bind_right  = false;
	filter->bind_top    = false;
	filter->bind_bottom = false;

	if (filter->effect != NULL) {
		obs_enter_graphics();
		gs_effect_destroy(filter->effect);
		filter->effect = NULL;
		obs_leave_graphics();
	}

	/* Load text */
	char *shader_text = NULL;

	const char *file_name =
			obs_data_get_string(filter->settings, "shader_file_name");

	/* Load default effect text if no file is selected */
	if (file_name && file_name[0] != '\0')
		shader_text = os_quick_read_utf8_file(file_name);
	else
		return;
	//shader_text = bstrdup(effect_template_default_image_shader);

	/* Load empty effect if file is empty / doesn't exist */
	if (shader_text == NULL)
		shader_text = bstrdup("");

	/* Create the effect. */
	char *errors = NULL;

	obs_enter_graphics();
	gs_effect_destroy(filter->effect);
	filter->effect = gs_effect_create(shader_text, NULL, &errors);
	obs_leave_graphics();

	bfree(shader_text);

	if (filter->effect == NULL) {
		blog(LOG_WARNING,
				"[obs-shader-filter] Unable to create effect. Errors returned from parser:\n%s",
				(errors == NULL || strlen(errors) == 0
								? "(None)"
								: errors));
	}
	bfree(errors);

	/* Prepare properties for mathmatical expressions to use */
	da_init(filter->vars);

	prep_te_funcs(&filter->vars.da);

	te_variable px_bind[] = {{"elapsed_time", &filter->elapsed_time_bind.f},
			{"uv_scale_x", &filter->uv_scale_bind.x},
			{"uv_scale_y", &filter->uv_scale_bind.y},
			{"uv_pixel_interval_x",
					&filter->uv_pixel_interval_bind.x},
			{"uv_pixel_interval_y",
					&filter->uv_pixel_interval_bind.y}};

	darray_push_back_array(
			sizeof(te_variable), &filter->vars.da, &px_bind[0], 5);

	/* Store references to the new effect's parameters. */
	da_init(filter->stored_param_list);
	size_t effect_count = gs_effect_get_num_params(filter->effect);
	for (i = 0; i < effect_count; i++) {
		gs_eparam_t *param =
				gs_effect_get_param_by_idx(filter->effect, i);
		update_filter_cache(filter, param);
	}
}

static const char *shader_filter_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("ShaderFilter");
}

static void *shader_filter_create(obs_data_t *settings, obs_source_t *source)
{
	UNUSED_PARAMETER(source);

	struct shader_filter_data *filter =
			bzalloc(sizeof(struct shader_filter_data));

	filter->context       = source;
	filter->settings      = settings;
	filter->reload_effect = true;
	filter->effect        = NULL;

	for (size_t i = 0; i < 4; i++)
		dstr_init(&filter->expr[i]);

	dstr_init(&filter->last_path);
	dstr_copy(&filter->last_path,
			obs_data_get_string(settings, "shader_file_name"));

	da_init(filter->stored_param_list);
	da_init(filter->eval_param_list);
	da_init(filter->vars);

	obs_source_update(source, settings);

	return filter;
}

static void shader_filter_destroy(void *data)
{
	struct shader_filter_data *filter = data;

	obs_data_release(filter->settings);
	dstr_free(&filter->last_path);
	obs_data_release(filter->settings);

	size_t i;
	for (i = 0; i < filter->stored_param_list.num; i++)
		effect_param_data_release(&filter->stored_param_list.array[i]);

	for (i = 0; i < 4; i++)
		dstr_free(&filter->expr[i]);

	obs_enter_graphics();
	gs_effect_destroy(filter->effect);
	filter->effect = NULL;
	obs_leave_graphics();

	da_free(filter->stored_param_list);
	da_free(filter->eval_param_list);
	da_free(filter->vars);

	bfree(filter);
}

static bool shader_filter_file_name_changed(obs_properties_t *props,
		obs_property_t *p, obs_data_t *settings)
{
	struct shader_filter_data *filter = obs_properties_get_param(props);
	const char *new_file_name =
			obs_data_get_string(settings, obs_property_name(p));

	if (dstr_is_empty(&filter->last_path) ||
			dstr_cmp(&filter->last_path, new_file_name) != 0) {
		filter->reload_effect = true;
		dstr_free(&filter->last_path);
		dstr_copy(&filter->last_path, new_file_name);
	}

	return true;
}

static bool shader_filter_reload_effect_clicked(
		obs_properties_t *props, obs_property_t *property, void *data)
{
	struct shader_filter_data *filter = data;

	filter->reload_effect = true;

	obs_source_update(filter->context, NULL);

	return true;
}

void set_expansion_bindings(gs_eparam_t *param, bool *bound_left,
		bool *bound_right, bool *bound_top, bool *bound_bottom)
{
	if (bound_left && !*bound_left &&
			get_eparam_bool(param, "bind_left", false)) {
		*bound_left = true;
	}
	if (bound_right && !*bound_right &&
			get_eparam_bool(param, "bind_right", false)) {
		*bound_right = true;
	}
	if (bound_top && !*bound_top &&
			get_eparam_bool(param, "bind_top", false)) {
		*bound_top = true;
	}
	if (bound_bottom && !bound_bottom &&
			get_eparam_bool(param, "bind_bottom", false)) {
		*bound_bottom = true;
	}
}

void set_expansion_bindings_vec(gs_eparam_t *param, bool *bound_left,
		bool *bound_right, bool *bound_top, bool *bound_bottom,
		size_t vec_num)
{
	bool *bound[4] = { bound_left, bound_right, bound_top, bound_bottom };
	const char *mixin = "xyzw";
	const char *bind_names[4] = { "bind_left", "bind_right", "bind_top",
			"bind_bottom" };
	struct dstr bind_name;
	dstr_init(&bind_name);
	size_t i;
	size_t j;
	for (i = 0; i < 4; i++) {
		if (bound[i] && !*bound[i]) {
			for (j = 0; j < vec_num; j++) {
				dstr_copy_cat(&bind_name, bind_names[i], "_",
						mixin + j, 1);
				if (get_eparam_bool(param, bind_name.array,
						false)) {
					*bound[i] = true;
					break;
				}
			}
		}
	}
	dstr_free(&bind_name);
}

void prep_bind_value(bool *bound, int *binding, struct effect_param_data *param,
		const char *name, bool is_float, struct dstr *expr,
		struct darray *var_list)
{

	if (bound && binding) {
		if (!*bound && get_eparam_bool(param->param, name, false)) {
			struct dstr formula_name;
			dstr_init_copy(&formula_name, name);
			dstr_cat(&formula_name, "_expr");

			char *expression = get_eparam_string(param->param,
					formula_name.array, param->name.array);
			dstr_free(&formula_name);

			dstr_free(expr);
			dstr_init_copy(expr, expression);
			bfree(expression);

			*bound = true;
		}
		/* update values */
		size_t i;
		const char *mixin = "xyzw";
		int vec_num = obs_get_vec_num(param->type);
		for (i = 0; i < vec_num; i++) {
			if (dstr_is_empty(&param->array_names[i])) {
				dstr_copy_cat(&param->array_names[i],
					param->name.array,
					"_", mixin + i, 1);
			}
		}

		switch (param->type) {
		case GS_SHADER_PARAM_BOOL:
		case GS_SHADER_PARAM_INT:
			param->te_bind.f[0] = (double)param->value.i;
			break;
		case GS_SHADER_PARAM_FLOAT:
			param->te_bind.f[0] = (double)param->value.f;
			break;
		case GS_SHADER_PARAM_INT2:
		case GS_SHADER_PARAM_INT3:
		case GS_SHADER_PARAM_INT4:
			for (i = 0; i < vec_num; i++)
				param->te_bind.f[i] =
						(double)param->value.l4.ptr[i];

			break;
		case GS_SHADER_PARAM_VEC2:
		case GS_SHADER_PARAM_VEC3:
		case GS_SHADER_PARAM_VEC4:
			for (i = 0; i < vec_num; i++)
				param->te_bind.f[i] =
						(double)param->value.v4.ptr[i];

			break;
		}
		/* bind values */
		if (!param->bound) {
			te_variable var[4] = {0};
			bool bind_array    = false;

			switch (param->type) {
			case GS_SHADER_PARAM_BOOL:
			case GS_SHADER_PARAM_FLOAT:
			case GS_SHADER_PARAM_INT:
				var[0].address = &param->te_bind.f[0];
				var[0].name    = param->name.array;
				break;
			case GS_SHADER_PARAM_INT2:
			case GS_SHADER_PARAM_INT3:
			case GS_SHADER_PARAM_INT4:
			case GS_SHADER_PARAM_VEC2:
			case GS_SHADER_PARAM_VEC3:
			case GS_SHADER_PARAM_VEC4:
				for (i = 0; i < vec_num; i++) {
					var[i].address = &param->te_bind.f[i];
					var[i].name    = param->array_names[i]
								      .array;
				}

				bind_array = true;
				break;
			default:
				return;
			}

			if (var[0].name == NULL) {
				printf("%s", var[0].name);
			}

			if (bind_array) {
				for (i = 0; i < vec_num; i++)
					append_te_variable(var_list, &var[i]);
			} else {
				append_te_variable(var_list, &var[0]);
			}

			param->bound = true;
		}
	}
	return;
}

void bind_compile(int *binding, te_variable *vars, const char *expression,
		int count)
{
	if (expression && strcmp(expression, "") != 0) {
		int err;
		te_expr *n = te_compile(expression, vars, count, &err);

		if (n) {
			*binding = (int)te_eval(n);
			te_free(n);
		} else {
			*binding = 0;
			blog(LOG_WARNING,
					"Error in expression: %.*s<< error "
					"here >>%s",
					err, expression, expression + err);
		}
	} else {
		*binding = 0;
	}
}

void bind_compile_float(float *binding, te_variable *vars,
		const char *expression, int count)
{
	if (expression && strcmp(expression, "") != 0) {
		int err;
		te_expr *n = te_compile(expression, vars, count, &err);

		if (n) {
			*binding = (float)te_eval(n);
			te_free(n);
		} else {
			*binding = 0;
			blog(LOG_WARNING,
				"Error in expression: %.*s<< error here >>%s",
				err, expression, expression + err);
		}
	} else {
		*binding = 0;
	}
}

void bind_compile_double(double *binding, te_variable *vars,
	const char *expression, int count)
{
	if (expression && strcmp(expression, "") != 0) {
		int err;
		te_expr *n = te_compile(expression, vars, count, &err);

		if (n) {
			*binding = te_eval(n);
			te_free(n);
		} else {
			*binding = 0;
			blog(LOG_WARNING,
				"Error in expression: %.*s<< error here >>%s",
				err, expression, expression + err);
		}
	} else {
		*binding = 0;
	}
}

void eval_param(struct effect_param_data *param,
		struct shader_filter_data *filter)
{
	size_t i;
	switch (param->type) {
	case GS_SHADER_PARAM_BOOL:
	case GS_SHADER_PARAM_INT:
		if (param->has_expr[0]) {
			param->value.l4.ptr[0] = 0;
			bind_compile(&param->value.l4.ptr[1], &filter->vars.array[0],
				param->expr[0].array,
				(int)filter->vars.num);
		}
		param->te_bind.f[0] = (double)param->value.i;
		break;
	case GS_SHADER_PARAM_FLOAT:
		if (param->has_expr[0]) {
			bind_compile_double(&param->value.f, &filter->vars.array[0],
				param->expr[0].array,
				(int)filter->vars.num);
		}
		param->te_bind.f[0] = param->value.f;
		break;
	case GS_SHADER_PARAM_INT2:
	case GS_SHADER_PARAM_INT3:
	case GS_SHADER_PARAM_INT4:
		for (i = 0; i < 4; i++) {
			if (param->has_expr[i]) {
				bind_compile(&param->value.l4.ptr[i],
						&filter->vars.array[0],
						param->expr[i].array,
						(int)filter->vars.num);
			}
			param->te_bind.f[i] =
				(double)param->value.l4.ptr[i];
		}
		break;
	case GS_SHADER_PARAM_VEC2:
	case GS_SHADER_PARAM_VEC3:
	case GS_SHADER_PARAM_VEC4:
		for (i = 0; i < 4; i++) {
			if (param->has_expr[i]) {
				bind_compile_float(&param->value.v4.ptr[i],
					&filter->vars.array[0],
					param->expr[i].array,
					(int)filter->vars.num);
			}
			param->te_bind.f[i] =
				(double)param->value.v4.ptr[i];
		}
		break;
	}
}

void prep_bind_values(bool *bound_left, bool *bound_right, bool *bound_top,
		bool *bound_bottom, struct effect_param_data *param,
		struct shader_filter_data *filter)
{

	int vec_num     = obs_get_vec_num(param->type);
	bool is_float   = (param->type == GS_SHADER_PARAM_FLOAT ||
                        param->type == GS_SHADER_PARAM_VEC2 ||
                        param->type == GS_SHADER_PARAM_VEC3 ||
                        param->type == GS_SHADER_PARAM_VEC4);
	bool *bounds[4] = {bound_left, bound_right, bound_top, bound_bottom};

	const char *bind_names[4] = {
			"bind_left", "bind_right", "bind_top", "bind_bottom"};
	const char *expr_names = "expr";
	const char *mixin = "xyzw";

	struct dstr bind_name;
	struct dstr expr_name;
	dstr_init(&bind_name);
	dstr_init(&expr_name);

	size_t i;
	size_t j;
	if (vec_num == 1) {
		dstr_free(&param->expr[0]);
		char *expr = get_eparam_string(param->param, expr_names, NULL);
		dstr_copy(&param->expr[0], expr);
		bfree(expr);
		if (!param->has_expr[0] && param->expr[0].array) {
			param->has_expr[0] = true;
			param->update_expr_per_frame[0] = get_eparam_bool(
					param->param, "update_expr_per_frame",
					false);
			if(param->update_expr_per_frame[0])
				param->update_per_frame = true;
			darray_push_back(sizeof(struct effect_param_data*),
				&filter->eval_param_list, &param);
		} else if (param->has_expr[0] && !param->expr[0].array) {
			darray_erase_item(sizeof(struct effect_param_data*),
				&filter->eval_param_list, &param);
			param->has_expr[0] = false;
		}

		for (i = 0; i < 4; i++) {
			prep_bind_value(bounds[i], &filter->resize.ptr[i],
					param, bind_names[i], is_float,
					&filter->expr[i], &filter->vars.da);
			if (!filter->bind_update_per_frame[i] && *bounds[i] &&
					((param->update_expr_per_frame[0] &&
					dstr_find(&filter->expr[i],
						param->name.array)) ||
					dstr_find(&filter->expr[i],
						"elapsed_time"))) {
				filter->bind_update_per_frame[i] = true;
			}
		}

		dstr_free(&expr_name);
		dstr_free(&bind_name);
		return;
	}

	for (j = 0; j < vec_num; j++) {
		dstr_copy_cat(&expr_name, expr_names, "_", mixin + j, 1);
		dstr_free(&param->expr[j]);
		char *expr = get_eparam_string(param->param, expr_name.array,
				NULL);
		dstr_copy(&param->expr[j], expr);
		bfree(expr);

		if (!param->has_expr[j] && param->expr[j].array) {
			for (i = 0; i < vec_num; i++)
				if (param->has_expr[i])
					break;
			/* we've never added this param */
			if (i >= vec_num) {
				darray_push_back(sizeof(struct effect_param_data*),
					&filter->eval_param_list, &param);
			}
			param->update_expr_per_frame[j] =
				get_eparam_bool(param->param,
					"update_expr_per_frame", false);
			if (param->update_expr_per_frame[j])
				param->update_per_frame = true;
			param->has_expr[j] = true;
		} else if (param->has_expr[j] && !param->expr[j].array) {
			param->has_expr[0] = false;
			for (i = 0; i < vec_num; i++)
				if (param->has_expr[i])
					break;
			param->update_expr_per_frame[j] = false;
			if (i >= vec_num) {
				darray_erase_item(sizeof(struct effect_param_data*),
					&filter->eval_param_list, &param);
			}
		}

		for (i = 0; i < 4; i++) {
			dstr_copy_cat(&bind_name, bind_names[i], "_", mixin + j,
					1);

			prep_bind_value(bounds[i], &filter->resize.ptr[i],
					param, bind_name.array, is_float,
					&filter->expr[i], &filter->vars.da);
			if (!filter->bind_update_per_frame[i] && *bounds[i] &&
				((param->update_expr_per_frame[i] &&
					dstr_find(&filter->expr[i],
						param->name.array)) ||
					dstr_find(&filter->expr[i],
						"elapsed_time"))) {
				filter->bind_update_per_frame[i] = true;
			}
		}
	}

	dstr_free(&expr_name);
	dstr_free(&bind_name);
}

void prep_param(struct shader_filter_data *filter,
		struct effect_param_data *param)
{
	prep_bind_values(&filter->bind_left, &filter->bind_right,
			&filter->bind_top, &filter->bind_bottom, param, filter);
}

void render_source(struct effect_param_data *param, float source_cx,
		float source_cy)
{

	uint32_t media_cx = obs_source_get_width(param->media_source);
	uint32_t media_cy = obs_source_get_height(param->media_source);

	if (!media_cx || !media_cy)
		return;

	float scale_x = source_cx / (float)media_cx;
	float scale_y = source_cy / (float)media_cy;

	gs_texrender_reset(param->texrender);
	if (gs_texrender_begin(param->texrender, media_cx, media_cy)) {
		struct vec4 clear_color;
		vec4_zero(&clear_color);

		gs_clear(GS_CLEAR_COLOR, &clear_color, 1, 0);
		gs_matrix_scale3f(scale_x, scale_y, 1.0f);
		obs_source_video_render(param->media_source);

		gs_texrender_end(param->texrender);
	} else {
		return;
	}

	gs_texture_t *tex = gs_texrender_get_texture(param->texrender);
	gs_effect_set_texture(param->param, tex);
}

void resize_audio_buffers(struct effect_param_data *param, size_t samples)
{
	if (param->max_sidechain_buf_frames < samples) {
		param->sidechain_buf = brealloc(param->sidechain_buf,
				samples * sizeof(float) * param->num_channels);

		param->max_sidechain_buf_frames = samples;
	}
}

static void sidechain_capture(void *p, obs_source_t *source,
		const struct audio_data *audio_data, bool muted)
{
	struct effect_param_data *param = p;

	UNUSED_PARAMETER(source);

	pthread_mutex_lock(&param->sidechain_mutex);

	if (param->max_sidechain_frames < audio_data->frames) {
		param->max_sidechain_frames = audio_data->frames;
		resize_audio_buffers(param, param->max_sidechain_frames);
	}

	size_t expected_size = param->max_sidechain_frames * sizeof(float);

	if (!expected_size)
		goto unlock;

	size_t i;
	if (param->sidechain_data[0].size > expected_size * 2) {
		for (i = 0; i < param->num_channels; i++) {
			circlebuf_pop_front(&param->sidechain_data[i], NULL,
					expected_size);
		}
	}

	if (muted) {
		for (i = 0; i < param->num_channels; i++) {
			circlebuf_push_back_zero(&param->sidechain_data[i],
					audio_data->frames * sizeof(float));
		}
	} else {
		for (i = 0; i < param->num_channels; i++) {
			circlebuf_push_back(&param->sidechain_data[i],
					audio_data->data[i],
					audio_data->frames * sizeof(float));
		}
	}

unlock:
	pthread_mutex_unlock(&param->sidechain_mutex);
}

void update_sidechain_callback(
		struct effect_param_data *param, const char *new_source_name)
{
	if (new_source_name) {
		obs_source_t *sidechain = NULL;
		if (new_source_name && *new_source_name)
			sidechain = obs_get_source_by_name(new_source_name);

		obs_source_t *old_sidechain = param->media_source;
		const char *old_sidechain_name =
				obs_source_get_name(old_sidechain);

		pthread_mutex_lock(&param->sidechain_update_mutex);
		if (old_sidechain) {
			/* Remove the current audio callback */
			obs_source_remove_audio_capture_callback(old_sidechain,
					sidechain_capture, param);
			/* Free up the source */
			obs_source_release(old_sidechain);

			/* Clear audio data */
			for (size_t i = 0; i < param->num_channels; i++) {
				circlebuf_pop_front(&param->sidechain_data[i],
						NULL,
						param->sidechain_data[i].size);
			}
		}
		/* Add the new audio callback */
		if (sidechain)
			obs_source_add_audio_capture_callback(
					sidechain, sidechain_capture, param);
		param->media_source = sidechain;
		pthread_mutex_unlock(&param->sidechain_update_mutex);
	}
}

static inline void get_sidechain_data(
		struct effect_param_data *param, const uint32_t num_samples)
{
	size_t data_size = num_samples * sizeof(float);
	if (!data_size)
		return;

	pthread_mutex_lock(&param->sidechain_mutex);
	if (param->max_sidechain_frames < num_samples)
		param->max_sidechain_frames = num_samples;

	if (param->sidechain_data[0].size < data_size) {
		pthread_mutex_unlock(&param->sidechain_mutex);
		goto clear;
	}

	for (size_t i = 0; i < param->num_channels; i++)
		circlebuf_peek_front(&param->sidechain_data[i],
			&param->sidechain_buf[num_samples * i],
			data_size);

	pthread_mutex_unlock(&param->sidechain_mutex);
	return;

clear:
	for (size_t i = 0; i < param->num_channels; i++)
		memset(&param->sidechain_buf[num_samples * i], 0, data_size);
}

static uint32_t process_audio(struct effect_param_data *param, size_t samples)
{
	size_t i;
	size_t j;
	size_t k;
	size_t h_samples     = samples / 2;
	size_t h_sample_size = samples * 2;

	for (i = 0; i < param->num_channels; i++) {
		audio_fft_complex(&param->sidechain_buf[i * samples],
				(uint32_t)samples);
	}
	for (i = 1; i < param->num_channels; i++) {
		memcpy(&param->sidechain_buf[i * h_samples],
				&param->sidechain_buf[i * samples],
				h_sample_size);
	}
	/* Calculate a periodogram */
	if (param->is_psd) {
		j = h_samples * param->num_channels;
		for (i = 0; i < j; i++) {
			param->sidechain_buf[i] = 10 * log10 ( (1 / (2.0 * M_PI_D * samples)) * pow(param->sidechain_buf[i], 2.0) );
		}
	}
	if (param->fft_bins < h_samples) {
		/* Take the average of the bins */
		size_t bin_width = h_samples / param->fft_bins;
		for (i = 0; i < param->num_channels; i++) {
			for (j = 0; j < param->fft_bins; j++) {
				float bin_sum = 0;
				for (k = 0; k < bin_width; k++) {
					bin_sum += param->sidechain_buf[k +
							i * h_samples +
							j * bin_width];
				}
				param->sidechain_buf[i * param->fft_bins + j] =
						bin_sum / bin_width;
			}
		}
		return param->fft_bins;
	}
	return (uint32_t)h_samples;
}

void get_window_function(struct effect_param_data *param)
{
	char *window = get_eparam_string(param->param, "window", NULL);
	param->window = get_window_type(window);
	bfree(window);
}

void render_audio_texture(struct effect_param_data *param, size_t samples)
{
	size_t px_width = samples;
	if (param->is_fft) {
		window_function(param->sidechain_buf, samples, param->window);
		px_width = process_audio(param, samples);
	}
	obs_enter_graphics();
	gs_texture_destroy(param->texture);
	param->texture = gs_texture_create((uint32_t)px_width,
			(uint32_t)param->num_channels, GS_R32F, 1,
			(const uint8_t **)&param->sidechain_buf, 0);
	obs_leave_graphics();
	gs_effect_set_texture(param->param, param->texture);
}

static const char *shader_filter_texture_file_filter =
		"Textures (*.bmp *.tga *.png *.jpeg *.jpg *.gif);;";

static const char *shader_filter_media_file_filter =
		"Video Files (*.mp4 *.ts *.mov *.wmv *.flv *.mkv *.avi *.gif *.webm);;";

static obs_properties_t *shader_filter_properties(void *data)
{
	struct shader_filter_data *filter = data;

	bool empty[4] = { 0 };
	struct dstr shaders_path = {0};
	dstr_init(&shaders_path);
	dstr_cat(&shaders_path, obs_get_module_data_path(obs_current_module()));
	dstr_cat(&shaders_path, "/shaders");

	obs_properties_t *props = obs_properties_create();

	obs_properties_set_param(props, filter, NULL);

	obs_properties_add_button(props, "reload_effect",
			obs_module_text("ShaderFilter.ReloadEffect"),
			shader_filter_reload_effect_clicked);

	obs_property_t *file_name = obs_properties_add_path(props,
			"shader_file_name",
			_MT("ShaderFilter.ShaderFileName"),
			OBS_PATH_FILE, NULL,
			shaders_path.array);

	obs_property_set_modified_callback(
			file_name, shader_filter_file_name_changed);

	obs_property_t *p  = NULL;
	size_t param_count = filter->stored_param_list.num;

	for (size_t param_index = 0; param_index < param_count; param_index++) {
		struct effect_param_data *param =
				(filter->stored_param_list.array + param_index);
		if (memcmp(&param->has_expr[0], &empty[0], sizeof(bool) * 4)
				!= 0)
			continue;

		const char *param_name = param->name.array;
		int i_tmp              = 0;
		char *c_tmp            = NULL;
		bool uses_module_text  = false;
		struct dstr n_param_name;
		struct dstr n_param_desc;
		dstr_init(&n_param_name);
		dstr_init(&n_param_desc);
		bool is_slider       = false;
		bool is_vec4         = false;
		bool is_source       = false;
		bool is_audio_source = false;
		bool is_media        = false;
		bool hide_descs      = false;
		bool hide_all_descs  = false;

		/* defaults for most parameters */
		/* handles <...[int|float] min [=]...>*/
		float f_min = get_eparam_float(param->param, "min", -FLT_MAX);
		int i_min   = get_eparam_int(param->param, "min", INT_MIN);

		/* handles <...[int|float] max [=]...>*/
		float f_max = get_eparam_float(param->param, "max", FLT_MAX);
		int i_max   = get_eparam_int(param->param, "max", INT_MAX);

		/* handles <...[int|float] step [=];...>*/
		float f_step = get_eparam_float(param->param, "step", 1);
		int i_step   = get_eparam_int(param->param, "step", 1);

		/* handles <...bool module_text [= true|false];...>*/
		uses_module_text = get_eparam_bool(
				param->param, "module_text", false);

		/* handles <...string name [= '...'|= "..."];...>*/
		char *desc = get_eparam_string(
				param->param, "name", param_name);
		if (desc) {
			dstr_free(&param->desc);
			if (uses_module_text)
				dstr_copy(&param->desc, desc);
			else
				dstr_copy(&param->desc, _MT(desc));
			bfree(desc);
		} else {
			dstr_free(&param->desc);
			if (uses_module_text)
				dstr_copy_dstr(&param->desc, &param->name);
			else
				dstr_copy(&param->desc, _MT(param->name.array));
		}

		hide_all_descs = get_eparam_bool(
				param->param, "hide_all_descs", false);
		hide_descs = get_eparam_bool(param->param, "hide_descs", false);
		obs_data_t *list_data = NULL;

		const char *param_desc =
				!hide_all_descs ? param->desc.array : NULL;

		int vec_num  = 1;
		bool is_list = get_eparam_bool(param->param, "is_list", false);
		bool is_float;

		switch (param->type) {
		case GS_SHADER_PARAM_BOOL:
			if (is_list) {
				p = obs_properties_add_list(props, param_name,
						param_desc, OBS_COMBO_TYPE_LIST,
						OBS_COMBO_FORMAT_INT);

				i_tmp = (int)get_eparam_bool(param->param,
						"enabled_module_text", false);
				c_tmp = get_eparam_string(param->param,
						"enabled_string", "On");
				obs_property_list_add_int(p,
						i_tmp ? _MT(c_tmp) : c_tmp, 1);
				bfree(c_tmp);

				i_tmp = (int)get_eparam_bool(param->param,
						"disabled_module_text", false);
				c_tmp = get_eparam_string(param->param,
						"disabled_string", "Off");
				obs_property_list_add_int(p,
						i_tmp ? _MT(c_tmp) : c_tmp, 0);
				bfree(c_tmp);
			} else {
				obs_properties_add_bool(
						props, param_name, param_desc);
			}
			break;
		case GS_SHADER_PARAM_FLOAT:
		case GS_SHADER_PARAM_INT:
			is_float = param->type == GS_SHADER_PARAM_FLOAT;
			is_slider     = get_eparam_bool(
                                        param->param, "is_slider", false);

			if (is_list) {
				p = obs_properties_add_list(props, param_name,
						param_desc, OBS_COMBO_TYPE_LIST,
						is_float ? OBS_COMBO_FORMAT_FLOAT
							 : OBS_COMBO_FORMAT_INT);

				if (is_float)
					fill_float_list(p, param->param);
				else
					fill_int_list(p, param->param);
			} else {
				if (is_float)
					obs_properties_add_float_prop(props,
							param_name, param_desc,
							f_min, f_max, f_step,
							is_slider);
				else
					obs_properties_add_int_prop(props,
							param_name, param_desc,
							i_min, i_max, i_step,
							is_slider);
			}

			break;
		case GS_SHADER_PARAM_INT2:
		case GS_SHADER_PARAM_INT3:
		case GS_SHADER_PARAM_INT4:
			vec_num = obs_get_vec_num(param->type);

			is_slider = get_eparam_bool(
					param->param, "is_slider", false);

			obs_properties_add_int_array(props, param_name,
					param_desc, i_min, i_max, i_step,
					is_slider, vec_num);

			break;
		case GS_SHADER_PARAM_VEC2:
		case GS_SHADER_PARAM_VEC3:
		case GS_SHADER_PARAM_VEC4:
			vec_num = obs_get_vec_num(param->type);

			is_vec4 = param->type == GS_SHADER_PARAM_VEC4 &&
					get_eparam_bool(param->param,
							"is_float4", false);

			if (!is_vec4 && param->type == GS_SHADER_PARAM_VEC4) {
				obs_properties_add_color(
						props, param_name, param_desc);
				break;
			}
			is_slider = get_eparam_bool(
					param->param, "is_slider", false);

			obs_properties_add_vec_array(props, param_name,
					param_desc, f_min, f_max, f_step,
					is_slider, vec_num);

			break;
		case GS_SHADER_PARAM_TEXTURE:
			is_source = get_eparam_bool(
					param->param, "is_source", false);
			if (is_source) {
				p = obs_properties_add_list(props, param_name,
						param_desc, OBS_COMBO_TYPE_LIST,
						OBS_COMBO_FORMAT_STRING);
				fill_source_list(p);
				break;
			}
			is_audio_source = get_eparam_bool(
					param->param, "is_audio_source", false);
			if (is_audio_source) {
				p = obs_properties_add_list(props, param_name,
						param_desc, OBS_COMBO_TYPE_LIST,
						OBS_COMBO_FORMAT_STRING);
				fill_audio_source_list(p);
				break;
			}

			is_media = get_eparam_bool(
					param->param, "is_media", false);
			if (is_media) {
				obs_properties_add_path(props, param_name,
						param_desc, OBS_PATH_FILE,
						shader_filter_media_file_filter,
						NULL);
				break;
			}

			obs_properties_add_path(props, param_name, param_desc,
					OBS_PATH_FILE,
					shader_filter_texture_file_filter,
					NULL);
			break;
		}

		param->is_vec4         = is_vec4;
		param->is_list         = is_list;
		param->is_source       = is_source;
		param->is_audio_source = is_audio_source;
		param->is_media        = is_media;

		dstr_free(&n_param_name);
		dstr_free(&n_param_desc);
	}

	dstr_free(&shaders_path);

	return props;
}

#define __DEBUG_GRAPHICS 0

void update_graphics_paramters(
		struct effect_param_data *param, float cx, float cy)
{
	struct vec4 color;
	switch (param->type) {
	case GS_SHADER_PARAM_FLOAT:
		gs_effect_set_float(param->param, (float)param->value.f);
#if __DEBUG_GRAPHICS
		blog(LOG_DEBUG, "%s: %f", param->name.array, param->value.f);
#endif
		break;
	case GS_SHADER_PARAM_BOOL:
	case GS_SHADER_PARAM_INT:
		gs_effect_set_int(param->param, (int)param->value.i);
#if __DEBUG_GRAPHICS
		blog(LOG_DEBUG, "%s: %d", param->name.array, param->value.i);
#endif
		break;
	case GS_SHADER_PARAM_INT2:
		gs_effect_set_vec2(
				param->param, (struct vec2 *)&param->value.l4);
#if __DEBUG_GRAPHICS
		blog(LOG_DEBUG, "%s.x: %d", param->name.array, param->value.l4.x);
		blog(LOG_DEBUG, "%s.y: %d", param->name.array, param->value.l4.y);
#endif
		break;
	case GS_SHADER_PARAM_INT3:
		gs_effect_set_vec3(
				param->param, (struct vec3 *)&param->value.l4);
#if __DEBUG_GRAPHICS
		blog(LOG_DEBUG, "%s.x: %d", param->name.array, param->value.l4.x);
		blog(LOG_DEBUG, "%s.y: %d", param->name.array, param->value.l4.y);
		blog(LOG_INFO, "%s.z: %d", param->name.array, param->value.l4.z);
#endif
		break;
	case GS_SHADER_PARAM_INT4:
		gs_effect_set_vec4(
				param->param, (struct vec4 *)&param->value.l4);
#if __DEBUG_GRAPHICS
		blog(LOG_INFO, "%s.x: %d", param->name.array, param->value.l4.x);
		blog(LOG_INFO, "%s.y: %d", param->name.array, param->value.l4.y);
		blog(LOG_INFO, "%s.z: %d", param->name.array, param->value.l4.z);
		blog(LOG_INFO, "%s.w: %d", param->name.array, param->value.l4.w);
#endif
		break;
	case GS_SHADER_PARAM_VEC2:
		gs_effect_set_vec2(
				param->param, (struct vec2 *)&param->value.v4);
#if __DEBUG_GRAPHICS
		blog(LOG_INFO, "%s.x: %f", param->name.array, param->value.v4.x);
		blog(LOG_INFO, "%s.y: %f", param->name.array, param->value.v4.y);
#endif
		break;
	case GS_SHADER_PARAM_VEC3:
		gs_effect_set_vec3(
				param->param, (struct vec3 *)&param->value.v4);
#if __DEBUG_GRAPHICS
		blog(LOG_INFO, "%s.x: %f", param->name.array, param->value.v4.x);
		blog(LOG_INFO, "%s.y: %f", param->name.array, param->value.v4.y);
		blog(LOG_INFO, "%s.z: %f", param->name.array, param->value.v4.z);
#endif
		break;
	case GS_SHADER_PARAM_VEC4:
		/* Treat as color or vec4 */
		if (param->is_vec4) {
			gs_effect_set_vec4(param->param,
					(struct vec4 *)&param->value.v4);
#if __DEBUG_GRAPHICS
			blog(LOG_INFO, "%s.x: %f", param->name.array, param->value.v4.x);
			blog(LOG_INFO, "%s.y: %f", param->name.array, param->value.v4.y);
			blog(LOG_INFO, "%s.z: %f", param->name.array, param->value.v4.z);
			blog(LOG_INFO, "%s.w: %f", param->name.array, param->value.v4.w);
#endif
		} else {
			vec4_from_rgba(&color, (unsigned int)param->value.i);
			gs_effect_set_vec4(param->param, &color);
#if __DEBUG_GRAPHICS
			blog(LOG_INFO, "%s.x: %f", param->name.array, color.x);
			blog(LOG_INFO, "%s.y: %f", param->name.array, color.y);
			blog(LOG_INFO, "%s.z: %f", param->name.array, color.z);
			blog(LOG_INFO, "%s.w: %f", param->name.array, color.w);
#endif
		}
		break;
	case GS_SHADER_PARAM_TEXTURE:
		/* Render texture from a source */
		if (param->is_source || param->is_media) {
			render_source(param, cx, cy);
			break;
		}

		if (param->is_audio_source) {
			resize_audio_buffers(param, AUDIO_OUTPUT_FRAMES);
			get_sidechain_data(param, param->fft_samples);
			render_audio_texture(param, param->fft_samples);
			break;
		}
		/* Otherwise use image file as texture */
		gs_effect_set_texture(param->param,
				(param->image ? param->image->texture : NULL));
		break;
	}
}

void get_graphics_parameters(struct effect_param_data *param,
		struct shader_filter_data *filter, obs_data_t *settings)
{
	const char *param_name = param->name.array;
	bool empty[4] = { 0 };
	if (memcmp(&param->has_expr[0], &empty[0], sizeof(bool) * 4) != 0)
		return;
	int vec_num;
	/* assign the value of the parameter from the properties */
	/* we take advantage of doing this step to "cache" values */
	param->update_per_frame = false;
	switch (param->type) {
	case GS_SHADER_PARAM_BOOL:
		if (param->is_list)
			param->value.i = obs_data_get_int(settings, param_name);
		else
			param->value.i =
					obs_data_get_bool(settings, param_name);

		prep_param(filter, param);
		break;
	case GS_SHADER_PARAM_FLOAT:
		param->value.f = (float)obs_data_get_double(
				settings, param_name);

		prep_param(filter, param);
		break;
	case GS_SHADER_PARAM_INT:
		param->value.i = (int)obs_data_get_int(settings, param_name);

		prep_param(filter, param);
		break;
	case GS_SHADER_PARAM_INT2:
	case GS_SHADER_PARAM_INT3:
	case GS_SHADER_PARAM_INT4:
		vec_num = obs_get_vec_num(param->type);
		for (size_t i = 0; i < vec_num; i++) {
			param->value.l4.ptr[i] = (int)obs_data_get_int(
					settings, param->array_names[i].array);
		}

		prep_param(filter, param);
		break;
	case GS_SHADER_PARAM_VEC2:
	case GS_SHADER_PARAM_VEC3:
		vec_num = obs_get_vec_num(param->type);
		for (size_t i = 0; i < vec_num; i++) {
			param->value.v4.ptr[i] = (float)obs_data_get_double(
					settings, param->array_names[i].array);
		}

		prep_param(filter, param);
		break;
	case GS_SHADER_PARAM_VEC4:
		param->is_vec4 = get_eparam_bool(
				param->param, "is_float4", false);

		if (!param->is_vec4) {
			obs_data_set_default_int(
					settings, param_name, 0xff000000);
			param->value.i = obs_data_get_int(settings, param_name);
			break;
		}
		vec_num = obs_get_vec_num(param->type);
		for (size_t i = 0; i < vec_num; i++) {
			param->value.v4.ptr[i] = (float)obs_data_get_double(
					settings, param->array_names[i].array);
		}

		prep_param(filter, param);
		break;
	case GS_SHADER_PARAM_TEXTURE:
		param->is_source = get_eparam_bool(
				param->param, "is_source", false);
		param->update_per_frame = true;

		if (param->is_source) {
			if (!param->texrender)
				param->texrender = gs_texrender_create(
						GS_RGBA, GS_ZS_NONE);

			obs_source_release(param->media_source);
			param->media_source = obs_get_source_by_name(
					obs_data_get_string(
							settings, param_name));
			break;
		}

		param->is_audio_source = get_eparam_bool(
				param->param, "is_audio_source", false);

		if (param->is_audio_source) {
			param->is_fft = get_eparam_bool(
					param->param, "is_fft", false);
			if (param->is_fft) {
				param->fft_samples = hlsl_clamp(get_eparam_int(
						param->param, "fft_samples",
						1024), 64, 1024);
				param->is_psd = get_eparam_bool(param->param,
						"is_psd", false);
				param->fft_bins = get_eparam_int(
						param->param, "fft_bins", 512);
				get_window_function(param);
			}
			const char *source_name = obs_data_get_string(
					settings, param_name);
			update_sidechain_callback(param, source_name);

			break;
		}

		param->is_media = get_eparam_bool(
				param->param, "is_media", false);

		if (param->is_media) {
			if (!param->texrender)
				param->texrender = gs_texrender_create(
						GS_RGBA, GS_ZS_NONE);
			const char *path = obs_data_get_string(
					settings, param_name);

			obs_data_t *media_settings = obs_data_create();
			obs_data_set_string(media_settings, "local_file", path);

			obs_source_release(param->media_source);
			param->media_source = obs_source_create_private(
					"ffmpeg_source", NULL, media_settings);

			obs_data_release(media_settings);
			break;
		}

		if (param->image == NULL) {
			param->image = bzalloc(sizeof(gs_image_file_t));
		} else {
			obs_enter_graphics();
			gs_image_file_free(param->image);
			obs_leave_graphics();
		}
		gs_image_file_init(param->image,
				obs_data_get_string(settings, param_name));

		obs_enter_graphics();
		gs_image_file_init_texture(param->image);
		obs_leave_graphics();
		break;
	}
}

static void shader_filter_update(void *data, obs_data_t *settings)
{
	struct shader_filter_data *filter = data;

	if (filter->reload_effect) {
		filter->reload_effect = false;

		shader_filter_reload_effect(filter);
		obs_source_update_properties(filter->context);
	}

	const size_t num_channels = audio_output_get_channels(obs_get_audio());

	float src_cx = (float)obs_source_get_width(filter->context);
	float src_cy = (float)obs_source_get_height(filter->context);

	const char *mixin  = "xyzw";
	size_t param_count = filter->stored_param_list.num;
	size_t i;
	for (size_t param_i = 0; param_i < param_count; param_i++) {
		struct effect_param_data *param =
				(filter->stored_param_list.array + param_i);
		const char *param_name = param->name.array;
		param->num_channels = num_channels;

		/* get the property names (if this was meant to be an array) */
		for (i = 0; i < 4; i++) {
			if (dstr_is_empty(&param->array_names[i])) {
				dstr_copy_cat(&param->array_names[i],
						param_name, "_", mixin + i, 1);
			}
		}

		get_graphics_parameters(param, filter, settings);
	}

	/* Eval parameters */
	bool empty[4] = { 0 };
	size_t eval_count = filter->eval_param_list.num;
	for (i = 0; i < eval_count; i++) {
		struct effect_param_data *param =
			*(filter->eval_param_list.array + i);
		if (memcmp(&param->update_expr_per_frame[0], &empty[0],
				sizeof(bool)*4) == 0) {
			eval_param(param, filter);
		}
	}

	/* Single pass update values */
	for (i = 0; i < param_count; i++) {
		struct effect_param_data *param =
			(filter->stored_param_list.array + i);
		if (!param->update_per_frame)
			update_graphics_paramters(param, src_cx, src_cy);
	}

	/* Calculate the stretch of the size of the source via expression */
	for (i = 0; i < 4; i++) {
		if (!filter->bind_update_per_frame[i])
			bind_compile(&filter->resize.ptr[i],
				&filter->vars.array[0],
				filter->expr[i].array,
				(int)filter->vars.num);
	}

	/* Calculate expansions if not already set. */
	/* Will be used in the video_tick() callback. */
	if (!filter->bind_left)
		filter->resize_left = 0;

	if (!filter->bind_right)
		filter->resize_right = 0;

	if (!filter->bind_top)
		filter->resize_top = 0;

	if (!filter->bind_bottom)
		filter->resize_bottom = 0;
}

static void shader_filter_tick(void *data, float seconds)
{
	struct shader_filter_data *filter = (struct shader_filter_data *)data;
	obs_source_t *target = obs_filter_get_target(filter->context);

	bool empty[4] = { 0 };
	size_t i;
	size_t param_count = filter->eval_param_list.num;
	for (i = 0; i < param_count; i++) {
		struct effect_param_data *param =
			*(filter->eval_param_list.array + i);
		if (memcmp(&param->update_expr_per_frame[0], &empty[0],
			sizeof(bool) * 4) != 0) {
			eval_param(param, filter);
		}
	}

	/* Calculate the stretch of the size of the source via expression */
	for (i = 0; i < 4; i++) {
		if(filter->bind_update_per_frame[i])
			bind_compile(&filter->resize.ptr[i],
					&filter->vars.array[0],
					filter->expr[i].array,
					(int)filter->vars.num);
	}

	/* Determine offsets from expansion values. */
	int base_width  = obs_source_get_base_width(target);
	int base_height = obs_source_get_base_height(target);

	filter->total_width =
			filter->resize_left + base_width + filter->resize_right;
	filter->total_height = filter->resize_top + base_height +
			filter->resize_bottom;

	filter->uv_scale.x = (float)filter->total_width / base_width;
	filter->uv_scale.y = (float)filter->total_height / base_height;

	filter->uv_scale_bind.x = filter->uv_scale.x;
	filter->uv_scale_bind.y = filter->uv_scale.y;

	filter->uv_offset.x = (float)(-filter->resize_left) / base_width;
	filter->uv_offset.y = (float)(-filter->resize_top) / base_height;

	filter->uv_pixel_interval.x = 1.0f / base_width;
	filter->uv_pixel_interval.y = 1.0f / base_height;

	filter->uv_pixel_interval_bind.x = filter->uv_pixel_interval.x;
	filter->uv_pixel_interval_bind.y = filter->uv_pixel_interval.y;

	filter->elapsed_time += seconds;
	filter->elapsed_time_bind.f += seconds;
}

static void shader_filter_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);

	struct shader_filter_data *filter = data;

	if (filter->effect != NULL) {
		if (!obs_source_process_filter_begin(filter->context, GS_RGBA,
				    OBS_NO_DIRECT_RENDERING))
			return;

		if (filter->param_uv_scale != NULL)
			gs_effect_set_vec2(filter->param_uv_scale,
					&filter->uv_scale);

		if (filter->param_uv_offset != NULL)
			gs_effect_set_vec2(filter->param_uv_offset,
					&filter->uv_offset);

		if (filter->param_uv_pixel_interval != NULL)
			gs_effect_set_vec2(filter->param_uv_pixel_interval,
					&filter->uv_pixel_interval);

		if (filter->param_elapsed_time != NULL)
			gs_effect_set_float(filter->param_elapsed_time,
					filter->elapsed_time);

		float src_cx = (float)obs_source_get_width(filter->context);
		float src_cy = (float)obs_source_get_height(filter->context);

		/* Assign parameter values to filter */
		size_t param_count = filter->stored_param_list.num;
		for (size_t param_index = 0; param_index < param_count;
				param_index++) {
			struct effect_param_data *param = (param_index +
					filter->stored_param_list.array);
			if (param->update_per_frame)
				update_graphics_paramters(
						param, src_cx, src_cy);
		}

		obs_source_process_filter_end(filter->context, filter->effect,
				filter->total_width, filter->total_height);
	} else {
		obs_source_skip_video_filter(filter->context);
	}
}

static uint32_t shader_filter_getwidth(void *data)
{
	struct shader_filter_data *filter = data;

	return filter->total_width;
}

static uint32_t shader_filter_getheight(void *data)
{
	struct shader_filter_data *filter = data;

	return filter->total_height;
}

static void shader_filter_defaults(obs_data_t *settings)
{
}

struct obs_source_info shader_filter = {.id = "obs_shader_filter",
		.type                       = OBS_SOURCE_TYPE_FILTER,
		.output_flags               = OBS_SOURCE_VIDEO,
		.create                     = shader_filter_create,
		.destroy                    = shader_filter_destroy,
		.update                     = shader_filter_update,
		.video_tick                 = shader_filter_tick,
		.get_name                   = shader_filter_get_name,
		.get_defaults               = shader_filter_defaults,
		.get_width                  = shader_filter_getwidth,
		.get_height                 = shader_filter_getheight,
		.video_render               = shader_filter_render,
		.get_properties             = shader_filter_properties};

bool obs_module_load(void)
{
	obs_register_source(&shader_filter);

	struct obs_audio_info aoi;
	obs_get_audio_info(&aoi);
	sample_rate = (double)aoi.samples_per_sec;
	output_channels = (double)get_audio_channels(aoi.speakers);

	return true;
}

void obs_module_unload(void)
{
}
