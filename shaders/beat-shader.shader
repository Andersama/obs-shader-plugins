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
	AddressU  = Wrap;
	AddressV  = Border;
	BorderColor = 00000000;	
};

sampler_state textureSampler_V {
	Filter    = Linear;
	AddressU  = Border;
	AddressV  = Wrap;
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
#define BINS 64

uniform texture2d audio <bool is_audio_source = true; bool is_fft = true; int fft_bins = BINS;>;
uniform int strength <bool is_slider = true; int min = 0; int max = 1920;>;

float4 mainImage(VertData v_in) : TARGET
{
	//float2 sample_uv = float2(floor(v_in.uv.x * BINS) / BINS, v_in.uv.y);
	float2 sample_uv = float2(0, 0.25);
	float4 fft_sample = audio.Sample(textureSampler, sample_uv);
	float shift = sin(fft_sample.r);//sin(fft_sample.r);
	float4 color = image.Sample(textureSampler, v_in.uv + float2(0, shift * strength * uv_pixel_interval.y));
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

	/*
	float2 px = !vertical * v_in.uv + vertical * float2(v_in.uv.y, v_in.uv.x);
	float4 color = audio.Sample(textureSampler, px);
	float2 shift = !vertical * float2(1, 0) + vertical * float2(0, 1);
	float2 px_2 = shift * (sin(elapsed_time + color.r) * color.r * px_shift);
	*/
	//return float4(px_2.xy,0,1);
	//return float4(color.r, color.r, color.r, 1);
	//return lerp(quiet_color, loud_color, color.r);