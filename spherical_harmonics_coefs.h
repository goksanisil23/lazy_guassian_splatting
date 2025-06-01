#pragma once
//  More information about real spherical harmonics can be obtained from:
//  https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
//  https://github.com/NVlabs/tiny-cuda-nn/blob/master/scripts/gen_sh.py

constexpr float SH_C0_0 = 0.28209479177387814;

constexpr float SH_C1_0 = -0.4886025119029199;
constexpr float SH_C1_1 = 0.4886025119029199;
constexpr float SH_C1_2 = -0.4886025119029199;

constexpr float SH_C2_0 = 1.0925484305920792;
constexpr float SH_C2_1 = -1.0925484305920792;
constexpr float SH_C2_2 = 0.31539156525252005;
constexpr float SH_C2_3 = -1.0925484305920792;
constexpr float SH_C2_4 = 0.5462742152960396;

constexpr float SH_C3_0 = -0.5900435899266435;
constexpr float SH_C3_1 = 2.890611442640554;
constexpr float SH_C3_2 = -0.4570457994644658;
constexpr float SH_C3_3 = 0.3731763325901154;
constexpr float SH_C3_4 = -0.4570457994644658;
constexpr float SH_C3_5 = 1.445305721320277;
constexpr float SH_C3_6 = -0.5900435899266435;

constexpr float SH_C4_0 = 2.5033429417967046;
constexpr float SH_C4_1 = -1.7701307697799304;
constexpr float SH_C4_2 = 0.9461746957575601;
constexpr float SH_C4_3 = -0.6690465435572892;
constexpr float SH_C4_4 = 0.10578554691520431;
constexpr float SH_C4_5 = -0.6690465435572892;
constexpr float SH_C4_6 = 0.47308734787878004;
constexpr float SH_C4_7 = -1.7701307697799304;
constexpr float SH_C4_8 = 0.6258357354491761;

constexpr float SH_C5_0  = -0.65638205684017015;
constexpr float SH_C5_1  = 8.3026492595241645;
constexpr float SH_C5_2  = -0.48923829943525038;
constexpr float SH_C5_3  = 4.7935367849733241;
constexpr float SH_C5_4  = -0.45294665119569694;
constexpr float SH_C5_5  = 0.1169503224534236;
constexpr float SH_C5_6  = -0.45294665119569694;
constexpr float SH_C5_7  = 2.3967683924866621;
constexpr float SH_C5_8  = -0.48923829943525038;
constexpr float SH_C5_9  = 2.0756623148810411;
constexpr float SH_C5_10 = -0.65638205684017015;
