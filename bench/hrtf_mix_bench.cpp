/* Standalone benchmark: HRTF binaural filtering + mixing for many sources.
 *
 * Compares the original production kernels (frozen copies below) against the
 * optimized kernels in the repo (core/mixer/mixer_sse.cpp, mixer_avx2.cpp),
 * measuring the per-buffer cost of the "mix stage" (N x MixHrtf followed by
 * one MixDirectHrtf, mirroring Voice::DoMix + alu.cpp) at three source
 * counts, and verifying bit-exact (per-sample identical) output.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include <mmintrin.h>
#include <xmmintrin.h>

#include "core/bufferline.h"
#include "core/mixer/hrtfdefs.h"
#include "core/mixer/hrtfbase.h"
#include "core/mixer/defs.h"

namespace {

/* ------------------------------------------------------------------ */
/* Frozen copy of the ORIGINAL SSE ApplyCoeffs (pre-optimization),     */
/* used as the bit-exactness and timing baseline.                      */
/* ------------------------------------------------------------------ */

force_inline auto ref_vmadd(__m128 const x, __m128 const y, __m128 const z) noexcept -> __m128
{ return _mm_add_ps(x, _mm_mul_ps(y, z)); }

void ApplyCoeffs_SSE_orig(std::span<f32x2> const Values, std::size_t const IrSize,
    ConstHrirSpan const Coeffs, float const left, float const right) noexcept
{
    ASSUME(IrSize >= MinIrLength);
    ASSUME(IrSize <= HrirLength);
    auto const lrlr = _mm_setr_ps(left, right, left, right);
    auto const count4 = (IrSize+1) >> 1;

    if(!(reinterpret_cast<uintptr_t>(Values.data())&15))
    {
        auto const vals4 = std::span{reinterpret_cast<__m128*>(Values[0].data()), count4};
        auto const coeffs4 = std::span{reinterpret_cast<const __m128*>(Coeffs[0].data()), count4};

        std::ranges::transform(vals4, coeffs4, vals4.begin(),
            [lrlr](__m128 const &val, __m128 const &coeff) -> __m128
            { return ref_vmadd(val, coeff, lrlr); });
    }
    else
    {
        auto coeffs = _mm_load_ps(Coeffs[0].data());
        auto vals = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64*>(Values[0].data()));
        auto imp0 = _mm_mul_ps(lrlr, coeffs);
        vals = _mm_add_ps(imp0, vals);
        _mm_storel_pi(reinterpret_cast<__m64*>(Values[0].data()), vals);
        auto td = count4 - 1_uz;
        auto i = 1_uz;
        do {
            coeffs = _mm_load_ps(Coeffs[i+1].data());
            vals = _mm_load_ps(Values[i].data());
            auto const imp1 = _mm_mul_ps(lrlr, coeffs);
            imp0 = _mm_shuffle_ps(imp0, imp1, _MM_SHUFFLE(1, 0, 3, 2));
            vals = _mm_add_ps(imp0, vals);
            _mm_store_ps(Values[i].data(), vals);
            imp0 = imp1;
            i += 2;
        } while(--td);
        vals = _mm_loadl_pi(vals, reinterpret_cast<__m64*>(Values[i].data()));
        imp0 = _mm_movehl_ps(imp0, imp0);
        vals = _mm_add_ps(imp0, vals);
        _mm_storel_pi(reinterpret_cast<__m64*>(Values[i].data()), vals);
    }
}

void MixHrtf_SSE_orig(std::span<float const> const InSamples, std::span<f32x2> const AccumSamples,
    unsigned const IrSize, MixHrtfFilter const *const hrtfparams, std::size_t const SamplesToDo)
    noexcept
{ MixHrtfBase<ApplyCoeffs_SSE_orig>(InSamples, AccumSamples, IrSize, hrtfparams, SamplesToDo); }

void MixHrtfBlend_SSE_orig(std::span<float const> const InSamples,
    std::span<f32x2> const AccumSamples, unsigned const IrSize,
    HrtfFilter const *const oldparams, MixHrtfFilter const *const newparams,
    std::size_t const SamplesToDo) noexcept
{
    MixHrtfBlendBase<ApplyCoeffs_SSE_orig>(InSamples, AccumSamples, IrSize, oldparams, newparams,
        SamplesToDo);
}

void MixDirectHrtf_SSE_orig(FloatBufferSpan const LeftOut, FloatBufferSpan const RightOut,
    std::span<FloatBufferLine const> const InSamples, std::span<f32x2> const AccumSamples,
    std::span<float, BufferLineSize> const TempBuf, std::span<HrtfChannelState> const ChanState,
    std::size_t const IrSize, std::size_t const SamplesToDo) noexcept
{
    MixDirectHrtfBase<ApplyCoeffs_SSE_orig>(LeftOut, RightOut, InSamples, AccumSamples, TempBuf,
        ChanState, IrSize, SamplesToDo);
}

/* ------------------------------------------------------------------ */

struct XorShift {
    std::uint32_t s;
    auto next() -> std::uint32_t
    { s ^= s<<13; s ^= s>>17; s ^= s<<5; return s; }
    auto frand() -> float
    { return static_cast<float>(static_cast<double>(next()>>8) * (1.0/16777216.0) * 2.0 - 1.0); }
};

using MixHrtfFn = auto(*)(std::span<float const>, std::span<f32x2>, unsigned,
    MixHrtfFilter const*, std::size_t) noexcept -> void;
using MixHrtfBlendFn = auto(*)(std::span<float const>, std::span<f32x2>, unsigned,
    HrtfFilter const*, MixHrtfFilter const*, std::size_t) noexcept -> void;
using MixDirectHrtfFn = auto(*)(FloatBufferSpan, FloatBufferSpan,
    std::span<FloatBufferLine const>, std::span<f32x2>, std::span<float, BufferLineSize>,
    std::span<HrtfChannelState>, std::size_t, std::size_t) noexcept -> void;

struct Impl {
    std::string name;
    MixHrtfFn mix;
    MixHrtfBlendFn blend;
    MixDirectHrtfFn direct;
};

struct Source {
    alignas(16) HrirArray coeffs{};
    u32x2 delay{};
    float gain0{}, gain1{};
    std::array<float, HrtfHistoryLength + BufferLineSize> input{};
};

/* Everything the mix stage touches, so runs can be reproduced bit-exactly. */
struct Stage {
    alignas(16) std::array<f32x2, BufferLineSize+HrirLength> accum{};
    FloatBufferLine leftOut{}, rightOut{};
    std::array<FloatBufferLine, 2> directIn{};
    std::array<float, BufferLineSize> temp{};
    std::array<HrtfChannelState, 2> chanState{};
    std::vector<Source> sources;
};

auto make_stage(std::size_t const numSources, std::uint32_t const seed) -> Stage
{
    auto rng = XorShift{seed};
    auto stage = Stage{};
    stage.sources.resize(numSources);
    for(auto &s : stage.sources)
    {
        for(auto &c : s.coeffs) c = f32x2{{rng.frand()*0.5f, rng.frand()*0.5f}};
        s.delay = u32x2{{rng.next()%HrtfHistoryLength, rng.next()%HrtfHistoryLength}};
        s.gain0 = 0.5f + 0.25f*rng.frand();
        s.gain1 = 0.5f + 0.25f*rng.frand();
        for(auto &v : s.input) v = rng.frand();
    }
    for(auto &line : stage.directIn)
        for(auto &v : line) v = rng.frand();
    for(auto &cs : stage.chanState)
    {
        cs.mSplitter.init(0.25f);
        cs.mHfScale = 0.8f;
        for(auto &c : cs.mCoeffs) c = f32x2{{rng.frand()*0.5f, rng.frand()*0.5f}};
    }
    return stage;
}

void run_stage(Impl const &impl, Stage &stage, unsigned const irSize, int const buffers)
{
    auto const accum = std::span{stage.accum};
    auto const leftOut = std::span{stage.leftOut};
    auto const rightOut = std::span{stage.rightOut};
    for(auto b = 0;b < buffers;++b)
    {
        for(auto const &s : stage.sources)
        {
            auto const params = MixHrtfFilter{s.coeffs, s.delay, s.gain0,
                (s.gain1-s.gain0) / static_cast<float>(BufferLineSize)};
            impl.mix(s.input, accum, irSize, &params, BufferLineSize);
        }
        impl.direct(leftOut, rightOut, stage.directIn, accum, stage.temp, stage.chanState,
            irSize, BufferLineSize);
    }
}

auto snapshot(Stage const &stage) -> std::vector<float>
{
    auto out = std::vector<float>{};
    auto const add = [&out](std::span<float const> v)
    { out.insert(out.end(), v.begin(), v.end()); };
    add(std::span{reinterpret_cast<float const*>(stage.accum.data()), stage.accum.size()*2});
    add(stage.leftOut);
    add(stage.rightOut);
    return out;
}

auto bit_compare(std::vector<float> const &a, std::vector<float> const &b) -> bool
{ return a.size() == b.size()
    && std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0; }

void check_correctness(Impl const &ref, std::vector<Impl> const &impls, unsigned const irSize)
{
    constexpr auto numSources = 8_uz;
    constexpr auto buffers = 4;

    auto refStage = make_stage(numSources, 1234u);
    run_stage(ref, refStage, irSize, buffers);
    auto const refOut = snapshot(refStage);

    /* Also exercise the blend (crossfade) path. */
    auto rng = XorShift{99u};
    auto oldparams = HrtfFilter{};
    for(auto &c : oldparams.Coeffs) c = f32x2{{rng.frand(), rng.frand()}};
    oldparams.Delay = u32x2{{3u, 7u}};
    oldparams.Gain = 0.6f;
    auto newcoeffs = HrirArray{};
    for(auto &c : newcoeffs) c = f32x2{{rng.frand(), rng.frand()}};
    auto const newparams = MixHrtfFilter{newcoeffs, u32x2{{11u, 5u}}, 0.0f, 0.8f/256.0f};
    auto blendInput = std::array<float, HrtfHistoryLength+1+BufferLineSize>{};
    for(auto &v : blendInput) v = rng.frand();
    auto blendAccumRef = std::array<f32x2, BufferLineSize+HrirLength>{};
    ref.blend(blendInput, blendAccumRef, irSize, &oldparams, &newparams, 256u);

    for(auto const &impl : impls)
    {
        auto stage = make_stage(numSources, 1234u);
        run_stage(impl, stage, irSize, buffers);
        auto const okMain = bit_compare(refOut, snapshot(stage));

        auto blendAccum = std::array<f32x2, BufferLineSize+HrirLength>{};
        impl.blend(blendInput, blendAccum, irSize, &oldparams, &newparams, 256u);
        auto const okBlend = std::memcmp(blendAccumRef.data(), blendAccum.data(),
            sizeof(blendAccumRef)) == 0;

        std::printf("  %-16s IrSize=%-3u  mix+direct: %s   blend: %s\n", impl.name.c_str(),
            irSize, okMain ? "BIT-EXACT" : "MISMATCH", okBlend ? "BIT-EXACT" : "MISMATCH");
    }
}

} // namespace

int main()
{
    auto const impls = std::vector<Impl>{
        {"C", MixHrtf_C, MixHrtfBlend_C, MixDirectHrtf_C},
        {"SSE (new)", MixHrtf_SSE, MixHrtfBlend_SSE, MixDirectHrtf_SSE},
#if HAVE_AVX2
        {"AVX2 (new)", MixHrtf_AVX2, MixHrtfBlend_AVX2, MixDirectHrtf_AVX2},
#endif
#if HAVE_AVX512
        {"AVX512 (new)", MixHrtf_AVX512, MixHrtfBlend_AVX2, MixDirectHrtf_AVX2},
#endif
    };
    auto const baseline = Impl{"SSE (baseline)", MixHrtf_SSE_orig, MixHrtfBlend_SSE_orig,
        MixDirectHrtf_SSE_orig};

    std::puts("== Per-sample (bit-exact) verification vs baseline ==");
    for(auto const irSize : {64u, 50u})
        check_correctness(baseline, impls, irSize);

    std::puts("");
    std::puts("== Mix stage timing (N x MixHrtf + MixDirectHrtf per 1024-sample buffer) ==");

    auto perfImpls = std::vector<Impl>{baseline};
    perfImpls.insert(perfImpls.end(), impls.begin(), impls.end());

    constexpr auto irSize = 64u;
    for(auto const numSources : {16_uz, 64_uz, 256_uz})
    {
        /* Scale the buffer count so each measurement covers ~the same work. */
        auto const buffers = static_cast<int>(65536_uz / numSources);
        std::printf("-- %zu sources, %d buffers --\n", numSources, buffers);

        auto baseTime = 0.0;
        for(auto const &impl : perfImpls)
        {
            auto best = 1e30;
            for(auto rep = 0;rep < 3;++rep)
            {
                auto stage = make_stage(numSources, 42u);
                run_stage(impl, stage, irSize, 2); /* warmup */
                auto const t0 = std::chrono::steady_clock::now();
                run_stage(impl, stage, irSize, buffers);
                auto const t1 = std::chrono::steady_clock::now();
                best = std::min(best, std::chrono::duration<double>(t1-t0).count());
            }
            if(impl.name == baseline.name) baseTime = best;
            auto const usPerBuffer = best / static_cast<double>(buffers) * 1e6;
            std::printf("  %-16s %9.2f us/buffer   %6.2f us/source   vs baseline: %+.1f%%\n",
                impl.name.c_str(), usPerBuffer, usPerBuffer/static_cast<double>(numSources),
                (best/baseTime - 1.0)*100.0);
        }
    }
    return 0;
}
