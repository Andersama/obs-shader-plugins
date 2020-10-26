# [obs-shader-plugins](https://github.com/Andersama/obs-shader-plugins)
>Rapidly prototype and create graphical effects using OBS's shader syntax.

## Usage
>See [https://obsproject.com/docs/graphics.html](https://obsproject.com/docs/graphics.html) for the basics of OBS's shader syntax.
>
>This plugin makes use of annotations. Annotations are blocks of variables wrapped in `<>` which are used to describe the gui for the plugin. This plugin will read the file, extract the variables necessary for the shader to function and prepare a gui based on the ones given. Giving you the freedom to design the gui of your shader without needing to write any additional c/c++ code to update the shader's variables or the gui.

# Annotations
> This plugin makes use of a limited number of types made available in HLSL, namely `bool`, `int`, `float` and `string`. Annotation syntax as specified by Microsoft's HLSL is.
> ```c
> <DataType Name = Value; ... ;>
> ```
> For example to add a new integer parameter with a maximum value:
> ```c
> uniform int modes <int max = 5;>;
> ```
> Note where sensible annotations are cast to their appropriate typing constraints, for example this integer parameter could've used a float for it's maximum value, at the point of gui creation this value will be cast to an integer. Meaning for the most part `int`, `float` and `bool` annotations are exchangeable.

## Generic Annotations
> `[any type]`
> ### name
> ```c
> <string name;>
> ```
> This annotation determines the label text
> ### module_text
> ```c
> <bool module_text;>
> ```
> This annotation determines if the label text should be searched from OBS's ini files.

## Numerical Annotations
> `[int, int2, int3, int4, float, float2, float3, float4]`
> ### is_slider
> ```c
> <[bool] is_slider;>
> ```
> This boolean flag changes the gui from a numerical up/down combo box to a slider
> ### is_list
> ```c
> <[bool] is_list;>
> ```
> This boolean flag changes the gui into a drop down list (see [list syntax](#lists))
> ### min
> ```c
>  <[int,float,bool] min;>
> ```
> This annotation specifies the lower bound of a slider or combo box gui.
> ### max
> ```c
> <[int,float,bool] max;>
> ```
> This annotation specifies the upper bound of a slider or combo box gui.
> ### step
> ```c
> <[int,float,bool] step;>
> ```
> This annotation specifies the value that the combo box or slider will increment by.
> 
> ### Lists
> ```c
> <[int|float|bool] list_item = ?; string list_item_?_name = "">
> ```
> When `is_list == true` the gui will be changed to a drop down list. The values for the list are set following the example above. Any number of values can be specified, the `string list_item_?_name` determines the text that'll be shown to the user for that value. By default the text will assume the numerical value as the text to use. The values are loaded in from left to right into the drop down in top down order.

## Float4 Only
> `[float4]`
> ### is_float4
> ```c
> <bool is_float4;>
> ```
> The `[float4]` type is considered by default a four component rgba color ranging from 0-255 (tranformed into 0-1 range for the shader). `is_float4` set to true will treat float4 like all the other vectors.

## Texture Annotations
> `[texture2d]`
> ### is_source
> ```c
> <bool is_source;>
> ```
> This annotation will create a drop down list of active graphic sources that can be used as textures.
> ### is_audio_source
> ```c
> <bool is_audio_source;>
> ```
> This annotation will create a drop down list of active audio sources that can be used as textures.
> ### is_fft
> ```c
> <bool is_fft;>
> ```
> This annotation (in combination w/ an audio source) if set to true will perform an FFT on the audio data being recieved.

## Boolean Annotations
> `[bool]`
> ### is_list
> ```c
> <bool is_list;>
> ```
> This annotation if set to true changes the gui into a drop down list as opposed to a checkbox.
> 
> ### enabled_string
> ```c
> <string enabled_string;>
> ```
> This annotation specifies the text for the drop down list for its "true" value.
> ### enabled_module_text
> ```c
> <bool enabled_module_text;>
> ```
> This annotation determines whether the text should be searched from OBS's ini files.
> ### disabled_string
> ```c
> <string disabled_string;>
> ```
> This annotation specifies the text for the drop down list for it's "false" value.
> ### disabled_module_text
> ```c
> <bool disabled_module_text;>
> ```
> This annotation determines whether the text should be searched from OBS's ini files.

## Example Shader
This dead simple shader lets you mirror another source...in any source.
```c
uniform float4x4 ViewProj;
uniform texture2d image;

uniform float2 elapsed_time;
uniform float2 uv_offset;
uniform float2 uv_scale;
uniform float2 uv_pixel_interval;

sampler_state textureSampler {
	Filter    = Linear;
	AddressU  = Border;
	AddressV  = Border;
	BorderColor = 00000000;
};

struct VertData {
	float4 pos : POSITION;
	float2 uv  : TEXCOORD0;
};

VertData mainTransform(VertData v_in)
{
	VertData vert_out;
	vert_out.pos = mul(float4(v_in.pos.xyz, 1.0), ViewProj);
	vert_out.uv = v_in.uv * uv_scale + uv_offset;
	return vert_out;
}

uniform texture2d source <bool is_source = true;>;

float4 mainImage(VertData v_in) : TARGET
{    
    float4 color = source.Sample(textureSampler, float2(1-v_in.uv.x, v_in.uv.y));
	
    return color;
}

technique Draw
{
	pass p0
	{
		vertex_shader = mainTransform(v_in);
		pixel_shader = mainImage(v_in);
	}
}
```

## Advanced
> The ability to extract annotations also gives very intresting prospects. Here to give full flexibility of 
> the creative juices I've taken a handy library [tinyexpr](https://github.com/codeplea/tinyexpr) to give you the ability
> to write mathmatical expressions in order to evaluate parameters.
>
> Yes...that's right, if you can express the value that you want in a mathmatical formula. You can.
> Note: these values are not added to the gui, as they don't need to be shown.

## tinyexpr (& crop / expansion)
> `[bool, int, float]`
> ### expr
> ```c
> <string expr;>
> ```
> This annotation describes a mathmatical function to evaluate

> `[int2, int3, int4, float2, float3, float4]`
> ### expr_x, expr_y, expr_z, expr_w
> ```c
> <string expr_x; string expr_y; string expr_z; string expr_z;>
> ```
> These annotations describe a mathmatical expression to evalulate for computing each vector component.

> `[any of the above]`
> ### update_expr_per_frame
> ```c
> <bool update_expr_per_frame;>
> ```
> This annotation controls whether the expression is evaluated per frame
> ### Cropping / Expansion
> Note: Each direction is handled by one expression, the first expressions found will be considered the ones to evaulate, and are always evaulated per frame.
> These annotations specify mathmatical expressions to evaluate cropping / expansion of the frame in their respective directions by pixel amounts.
> Positive values = expansion, Negative = cropping.
> In addition they must be confirmed as bound `bool bind_left` `bool bind_right` `bool bind_top` `bool bind_bottom`

> `[bool, int, float, texture2d]`
> ### bind_left_expr, bind_right_expr, bind_top_expr, bind_bottom_expr
> ```c
> <string bind_left_expr; string bind_right_expr; string bind_top_expr; string bind_bottom_expr;>
> ```
> `[int2, int3, int4, float2, float3, float4]`

> For brevity's sake, these are like the above, but w/ their respective vector component like so:
> ```c
> <string bind_[direction]_[vector component]_expr;>
> ```

## Advanced Shader
```c
uniform float4x4 ViewProj;
uniform texture2d image;

uniform float elapsed_time;
uniform float2 uv_offset;
uniform float2 uv_scale;
uniform float2 uv_pixel_interval;

sampler_state textureSampler {
	Filter    = Linear;
	AddressU  = Border;
	AddressV  = Border;
	BorderColor = 00000000;
};

sampler_state textureSampler_H {
	Filter    = Linear;
	AddressU  = Border;
	AddressV  = Border;
	BorderColor = 00000000;	
};

sampler_state textureSampler_V {
	Filter    = Linear;
	AddressU  = Wrap;
	AddressV  = Border;
	BorderColor = 00000000;	
};

struct VertData {
	float4 pos : POSITION;
	float2 uv  : TEXCOORD0;
};

VertData mainTransform(VertData v_in)
{
	VertData vert_out;
	vert_out.pos = mul(float4(v_in.pos.xyz, 1.0), ViewProj);
	vert_out.uv = v_in.uv * uv_scale + uv_offset;
	return vert_out;
}

#define PI 3.141592653589793238462643383279502884197169399375105820974
#define PIO3 1.047197551196597746154214461093167628065723133125035273658
#define PI2O3 2.094395102393195492308428922186335256131446266250070547316

float melScale(float freq){
	return 2595 * log10 (1 + freq / 700.0);
}

float hertzFromMel(float mel) {
	return 700 * (pow(10, mel / 2595) - 1);
}

uniform texture2d audio <bool is_audio_source = true; bool is_fft = true; int fft_samples = 1024; string window = "blackmann_harris";>;
uniform bool vertical;
uniform float px_shift <bool is_slider = true; float min = 0.5; float max = 1920; float step = 0.5;>;
uniform bool show_fft;
uniform float sample_rate <string expr = "sample_rate";>;
uniform float mel_total <string expr = "mel_from_hz(sample_rate / 2)";>;
uniform float variability <bool is_slider = true; float min = 0; float max = 10000;>;
uniform float rand_num <string expr = "random(0,1.0)"; bool update_expr_per_frame = true;>;

float4 mainImage(VertData v_in) : TARGET
{
	float2 px;
	float mel;
	float hz;
	float4 color;
	float px_2;
	float2 shift;
	float db;
	if(vertical){
		px = float2((1 - distance(v_in.uv.y, 0.5) * 2), v_in.uv.x);
		color = audio.Sample(textureSampler_V, px + float2(elapsed_time * 0.0001 * variability * rand_num, 0));
		if(show_fft)
			return color;
		db = clamp( ((log10( 1 / (2 * PI * 1024) * pow(color.r,2) )) + 12) / 12.0, 0, 2 );
		px_2 = (sin(color.r) * db * px_shift);
		shift = float2(px_2 * uv_pixel_interval.x, 0);
		return image.Sample(textureSampler_H, v_in.uv + shift);
	} else {
		px = float2((1 - distance(v_in.uv.x, 0.5) * 2), v_in.uv.y);
		color = audio.Sample(textureSampler_V, px + float2(elapsed_time * 0.0001 * variability * rand_num, 0));
		if(show_fft)
			return color;
		db = clamp( ((log10( 1 / (2 * PI * 1024) * pow(color.r,2) )) + 12) / 12.0, 0, 2 );
		px_2 = (sin(color.r) * db * px_shift);
		shift = float2(0, px_2 * uv_pixel_interval.y);
		return image.Sample(textureSampler_V, v_in.uv + shift);
	}
}

technique Draw
{
	pass p0
	{
		vertex_shader = mainTransform(v_in);
		pixel_shader = mainImage(v_in);
	}
}
```

## Acknowledgments
> https://github.com/nleseul/obs-shaderfilter most of the underlying code was already hashed out by this wonderful plugin, this branch/plugin takes this plugin a few steps furthur.
