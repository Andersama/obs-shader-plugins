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
uniform float db_range <bool is_slider = true; float min = 10; float max = 120; float step = 0.25;>;

float4 mainImage(VertData v_in) : TARGET
{
	float2 px;
	float mel;
	float hz;
	float4 color;
	float px_2;
	float2 shift;
	float db;
	float2 rand_shift = float2(elapsed_time * 0.0001 * variability * rand_num, 0);
	float4 n_color;
	float _range = db_range / 10.0;
	if(vertical){
		px = float2((1 - distance(v_in.uv.y, 0.5) * 2), v_in.uv.y);
		color = audio.Sample(textureSampler_V, px + float2(elapsed_time * 0.0001 * variability * rand_num, 0));
		if(show_fft)
			return color;
		db = clamp( ((log10( 1 / (2 * PI * 1024) * pow(color.r,2) )) + _range) / _range, 0, 2 );
		px_2 = (sin(color.r) * db * px_shift);
		shift = float2(px_2 * uv_pixel_interval.x, 0);
		return image.Sample(textureSampler_H, v_in.uv + shift);
	} else {
		px = float2((1 - distance(v_in.uv.x, 0.5) * 2), v_in.uv.x);
		color = audio.Sample(textureSampler_V, px + rand_shift);
		if(show_fft)
			return color;
		db = clamp( ((log10( 1 / (2 * PI * 1024) * pow(color.r,2) )) + _range) / _range, 0, 2 );
		px_2 = (sin(color.r) * db * px_shift);
		shift = float2(0, px_2 * uv_pixel_interval.y);
		n_color.ra = image.Sample(textureSampler_V, v_in.uv + shift).ra;
		px_2 = (sin(color.r+PIO3) * db * px_shift);
		shift = float2(0, px_2 * uv_pixel_interval.y);
		n_color.g = image.Sample(textureSampler_V, v_in.uv + shift).g;
		px_2 = (sin(color.r+PI2O3) * db * px_shift);
		shift = float2(0, px_2 * uv_pixel_interval.y);
		n_color.b = image.Sample(textureSampler_V, v_in.uv + shift).b;
		return n_color;
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