#include "config.h"

#include <immintrin.h>

#include <cstddef>
#include <span>

#include "core/bufferline.h"
#include "core/mixer/hrtfdefs.h"
#include "defs.h"
#include "hrtfbase.h"
#include "opthelper_macros.hpp"


#if defined(__GNUC__) && !defined(__clang__) && !defined(__AVX2__) && !defined(__powerpc64__)
#pragma GCC target("avx2")
#endif

namespace {

/* Applies the HRIR coefficients to the accumulator, four IR taps (two
 * channels each) per 256-bit vector. Processes the exact same number of taps
 * as the SSE path ((IrSize+1)&~1) with the same per-element multiply-then-add
 * order and no FMA contraction, so results are bit-identical to the SSE and C
 * mixers. Unaligned accesses are used uniformly, removing the per-sample
 * alignment branch.
 */
void ApplyCoeffs(std::span<f32x2> const Values, std::size_t const IrSize,
    ConstHrirSpan const Coeffs, float const left, float const right) noexcept NONBLOCKING
{
    ASSUME(IrSize >= MinIrLength);
    ASSUME(IrSize <= HrirLength);
    auto const lr = _mm_setr_ps(left, right, left, right);
    auto const lrlr = _mm256_insertf128_ps(_mm256_castps128_ps256(lr), lr, 1);
    /* Round up the IR size to a multiple of 2 for SIMD (2 IRs for 2 channels
     * is 4 floats), matching the tap count of the SSE mixer. The underlying
     * HRIR is a fixed-size multiple of 2, any extra samples are either 0
     * (silence) or more IR samples that get applied for "free".
     */
    auto const count4 = (IrSize+1) >> 1;
    auto const count8 = count4 >> 1;

    auto *const valp = Values[0].data();
    auto const *const coefp = Coeffs[0].data();
    for(auto i = 0_uz;i < count8;++i)
    {
        auto const vals = _mm256_loadu_ps(valp + i*8);
        auto const coeffs = _mm256_loadu_ps(coefp + i*8);
        _mm256_storeu_ps(valp + i*8, _mm256_add_ps(vals, _mm256_mul_ps(coeffs, lrlr)));
    }
    if(count4&1)
    {
        auto const vals = _mm_loadu_ps(valp + count8*8);
        auto const coeffs = _mm_loadu_ps(coefp + count8*8);
        _mm_storeu_ps(valp + count8*8, _mm_add_ps(vals, _mm_mul_ps(coeffs, lr)));
    }
}

} // namespace

void MixHrtf_AVX2(std::span<float const> const InSamples, std::span<f32x2> const AccumSamples,
    unsigned const IrSize, MixHrtfFilter const *const hrtfparams, std::size_t const SamplesToDo)
    noexcept NONBLOCKING
{ MixHrtfBase<ApplyCoeffs>(InSamples, AccumSamples, IrSize, hrtfparams, SamplesToDo); }

void MixHrtfBlend_AVX2(std::span<float const> const InSamples,
    std::span<f32x2> const AccumSamples, unsigned const IrSize,
    HrtfFilter const *const oldparams, MixHrtfFilter const *const newparams,
    std::size_t const SamplesToDo) noexcept NONBLOCKING
{
    MixHrtfBlendBase<ApplyCoeffs>(InSamples, AccumSamples, IrSize, oldparams, newparams,
        SamplesToDo);
}

void MixDirectHrtf_AVX2(FloatBufferSpan const LeftOut, FloatBufferSpan const RightOut,
    std::span<FloatBufferLine const> const InSamples, std::span<f32x2> const AccumSamples,
    std::span<float, BufferLineSize> const TempBuf, std::span<HrtfChannelState> const ChanState,
    std::size_t const IrSize, std::size_t const SamplesToDo) noexcept NONBLOCKING
{
    MixDirectHrtfBase<ApplyCoeffs>(LeftOut, RightOut, InSamples, AccumSamples, TempBuf, ChanState,
        IrSize, SamplesToDo);
}
