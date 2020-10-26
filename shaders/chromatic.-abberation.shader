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

#define PI 3.141592653589793238462643383279502884197169399375105820974
#define PIO3 1.047197551196597746154214461093167628065723133125035273658
#define PI2O3 2.094395102393195492308428922186335256131446266250070547316

uniform float2 center <bool is_slider = true; int max = 1920; int min = -1920; int step = 1;>;

uniform int offset <bool is_slider = true; int max = 960; int min = -960; int step = 1;>;
uniform float angle <bool is_slider = true; float max = 360; float min = 0; float step = 0.5;>;

uniform bool square;
uniform bool cube;
uniform bool pixel_dist;

float4 mainImage(VertData v_in) : TARGET
{
	float avg = (uv_pixel_interval.x + uv_pixel_interval.y) / 2.0;
	float rad = radians(angle);
	float2 center_target = float2(0.5,0.5) + (center * uv_pixel_interval.xy);
	float2 dir = normalize(v_in.uv - center);
	float dist;
	if(pixel_dist) {
		dist = distance(v_in.uv / uv_pixel_interval.xy, center_target.xy / uv_pixel_interval.xy) * avg;
	} else {
		dist = distance(v_in.uv, center.xy) * avg;
	}
	float2 uv = offset * dir * dist * uv_pixel_interval;
	
	if(square){
		uv = uv * dist;
	} else if(cube) {
		uv = uv * dist * dist;
	}
	
    float2 uv_r = v_in.uv + (uv * ((sin(rad + PI2O3)+1.0)/2.0));
	float2 uv_g = v_in.uv + (uv * ((sin(rad + PIO3)+1.0)/2.0));
	float2 uv_b = v_in.uv + (uv * ((sin(rad)+1.0)/2.0));
    float4 color_r = image.Sample(textureSampler, uv_r);
    float4 color_g = image.Sample(textureSampler, uv_g);
	float4 color_b = image.Sample(textureSampler, uv_b);
	
	float4 color = float4(color_r.r, color_g.g, color_b.b, 1);
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