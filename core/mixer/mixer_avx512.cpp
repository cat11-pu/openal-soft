#include "config.h"

#include <immintrin.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>

#include "core/bufferline.h"
#include "core/mixer/hrtfdefs.h"
#include "defs.h"
#include "hrtfbase.h"
#include "opthelper_macros.hpp"


#if defined(__GNUC__) && !defined(__clang__) && !defined(__AVX512F__) && !defined(__powerpc64__)
#pragma GCC target("avx512f")
#endif

namespace {

/* Keep the compiler from contracting the multiply into the add: an FMA
 * rounds differently than the separate multiply+add the reference mixers
 * use, and results must stay bit-identical. MSVC doesn't contract explicit
 * intrinsics, so it needs no barrier.
 */
#if defined(_MSC_VER) && !defined(__clang__)
#define BLOCK_CONTRACTION(v) (static_cast<void>(0))
#else
#define BLOCK_CONTRACTION(v) asm("" : "+v"(v))
#endif

/* The HRTF mix accumulates each input sample into a sliding window of the
 * accumulator: accumulator element k receives exactly one multiply+add per
 * sample, in ascending sample order. That order must be preserved to keep
 * the output bit-identical, which rules out reassociation - but the window
 * itself (up to HrirLength f32x2 elements = 16 zmm registers) fits in
 * AVX-512's 32 registers. Keeping the window (and, for IR sizes up to 64,
 * the HRIR coefficients) register-resident reduces the per-sample memory
 * traffic to one 8-byte head store and one 8-byte tail load, avoiding the
 * store-to-load forwarding stalls of the streaming loop.
 *
 * NW is the window size in zmm registers (8 f32x2 elements each).
 */

force_inline auto shift1(__m512 const lo, __m512 const hi) noexcept -> __m512
{
    return _mm512_castsi512_ps(_mm512_alignr_epi64(_mm512_castps_si512(hi),
        _mm512_castps_si512(lo), 1));
}

/* std::array drops the alignment of __m512 (and warns about it); this keeps
 * it while still allowing scalar-replacement into registers.
 */
template<std::size_t N>
struct ZmmSet { __m512 m[N]; };

template<std::size_t NW, std::size_t... K>
void regwin_mac(ZmmSet<NW> &w, ZmmSet<NW> const &cregs,
    float const *const coefp, __m512 const lrlr, __mmask16 const lastmask,
    std::index_sequence<K...>) noexcept NONBLOCKING
{
    ([&]{
        auto const ck = [&]() -> __m512 {
            if constexpr(NW <= 8) return cregs.m[K];
            else return _mm512_loadu_ps(coefp + K*16);
        }();
        auto prod = _mm512_mul_ps(ck, lrlr);
        BLOCK_CONTRACTION(prod);
        if constexpr(K+1 < NW)
            w.m[K] = _mm512_add_ps(w.m[K], prod);
        else
            /* Apply exactly count4*2 taps like the other mixers; elements
             * past the tap count are left untouched, not even += 0.
             */
            w.m[K] = _mm512_mask_add_ps(w.m[K], lastmask, w.m[K], prod);
    }(), ...);
}

template<std::size_t NW, std::size_t... K>
void regwin_shift(ZmmSet<NW> &w, __m512 const tail,
    std::index_sequence<K...>) noexcept NONBLOCKING
{
    ((w.m[K] = shift1(w.m[K], (K+1 < NW) ? w.m[K+1] : tail)), ...);
}

template<std::size_t NW>
void RegWinMix(float const *const inL, float const *const inR, float const gain,
    float const gainStep, ConstHrirSpan const Coeffs, f32x2 *const accum,
    unsigned const count4, std::size_t const SamplesToDo) noexcept NONBLOCKING
{
    ASSUME(SamplesToDo > 0);
    ASSUME(SamplesToDo <= BufferLineSize);

    auto const *const coefp = Coeffs[0].data();
    auto cregs = ZmmSet<NW>{};
    if constexpr(NW <= 8)
    {
        for(auto k = 0_uz;k < NW;++k)
            cregs.m[k] = _mm512_loadu_ps(coefp + k*16);
    }

    auto *const accp = accum->data();
    auto w = ZmmSet<NW>{};
    for(auto k = 0_uz;k < NW;++k)
        w.m[k] = _mm512_loadu_ps(accp + k*16);

    /* Number of valid tap floats in the last window register (1..16). */
    auto const lastCount = count4*4u - static_cast<unsigned>((NW-1)*16);
    auto const lastmask = static_cast<__mmask16>((1u<<lastCount) - 1u);

    auto done = 0_uz;
    while(done < SamplesToDo)
    {
        auto const todo = std::min(std::size_t{256}, SamplesToDo-done);

        /* Batch pass: apply the gain ramp and per-ear delays for a chunk of
         * samples, computing the same values with the same rounding as the
         * reference per-sample loop. The ramp multiply and add are kept in
         * separate loops with a memory round-trip in between so the compiler
         * cannot contract them into an FMA (which would round differently);
         * both loops still auto-vectorize.
         */
        auto stepmuls = std::array<float, 256>{};
        for(auto i = 0_uz;i < todo;++i)
            stepmuls[i] = gainStep*static_cast<float>(done+i);
        auto pairs = std::array<f32x2, 256>{};
        for(auto i = 0_uz;i < todo;++i)
        {
            auto const g = gain + stepmuls[i];
            pairs[i] = f32x2{{inL[done+i]*g, inR[done+i]*g}};
        }

        for(auto i = 0_uz;i < todo;++i)
        {
            auto pairbits = std::uint64_t{};
            std::memcpy(&pairbits, &pairs[i], 8);
            auto const lrlr = _mm512_castsi512_ps(_mm512_set1_epi64(
                static_cast<long long>(pairbits)));

            regwin_mac<NW>(w, cregs, coefp, lrlr, lastmask,
                std::make_index_sequence<NW>{});

            /* The head element is final after this sample: store it, pull in
             * the new tail element, and shift the window by one f32x2.
             */
            auto const head = _mm_cvtsi128_si64(_mm512_castsi512_si128(
                _mm512_castps_si512(w.m[0])));
            std::memcpy(accp + (done+i)*2, &head, 8);
            auto tailbits = std::uint64_t{};
            std::memcpy(&tailbits, accp + (done+i+NW*8)*2, 8);
            auto const tail = _mm512_castsi512_ps(_mm512_zextsi128_si512(
                _mm_cvtsi64_si128(static_cast<long long>(tailbits))));
            regwin_shift<NW>(w, tail, std::make_index_sequence<NW>{});
        }
        done += todo;
    }

    /* Write back the in-flight part of the window. */
    for(auto k = 0_uz;k < NW;++k)
        _mm512_storeu_ps(accp + SamplesToDo*2 + k*16, w.m[k]);
}

} // namespace

void MixHrtf_AVX512(std::span<float const> const InSamples,
    std::span<f32x2> const AccumSamples, unsigned const IrSize,
    MixHrtfFilter const *const hrtfparams, std::size_t const SamplesToDo) noexcept NONBLOCKING
{
    ASSUME(IrSize >= MinIrLength);
    ASSUME(IrSize <= HrirLength);

    auto const count4 = (IrSize+1u) >> 1;
    auto const ldelay = std::size_t{HrtfHistoryLength} - hrtfparams->Delay[0];
    auto const rdelay = std::size_t{HrtfHistoryLength} - hrtfparams->Delay[1];
    auto const *const inL = InSamples.data() + ldelay;
    auto const *const inR = InSamples.data() + rdelay;
    auto const gain = hrtfparams->Gain;
    auto const gainStep = hrtfparams->GainStep;
    auto const coeffs = ConstHrirSpan{hrtfparams->Coeffs};
    auto *const accum = AccumSamples.data();

    switch((count4+3u) >> 2)
    {
    case 1: RegWinMix<1>(inL, inR, gain, gainStep, coeffs, accum, count4, SamplesToDo); break;
    case 2: RegWinMix<2>(inL, inR, gain, gainStep, coeffs, accum, count4, SamplesToDo); break;
    case 3: RegWinMix<3>(inL, inR, gain, gainStep, coeffs, accum, count4, SamplesToDo); break;
    case 4: RegWinMix<4>(inL, inR, gain, gainStep, coeffs, accum, count4, SamplesToDo); break;
    case 5: RegWinMix<5>(inL, inR, gain, gainStep, coeffs, accum, count4, SamplesToDo); break;
    case 6: RegWinMix<6>(inL, inR, gain, gainStep, coeffs, accum, count4, SamplesToDo); break;
    case 7: RegWinMix<7>(inL, inR, gain, gainStep, coeffs, accum, count4, SamplesToDo); break;
    case 8: RegWinMix<8>(inL, inR, gain, gainStep, coeffs, accum, count4, SamplesToDo); break;
    case 9: RegWinMix<9>(inL, inR, gain, gainStep, coeffs, accum, count4, SamplesToDo); break;
    case 10: RegWinMix<10>(inL, inR, gain, gainStep, coeffs, accum, count4, SamplesToDo); break;
    case 11: RegWinMix<11>(inL, inR, gain, gainStep, coeffs, accum, count4, SamplesToDo); break;
    case 12: RegWinMix<12>(inL, inR, gain, gainStep, coeffs, accum, count4, SamplesToDo); break;
    case 13: RegWinMix<13>(inL, inR, gain, gainStep, coeffs, accum, count4, SamplesToDo); break;
    case 14: RegWinMix<14>(inL, inR, gain, gainStep, coeffs, accum, count4, SamplesToDo); break;
    case 15: RegWinMix<15>(inL, inR, gain, gainStep, coeffs, accum, count4, SamplesToDo); break;
    case 16: RegWinMix<16>(inL, inR, gain, gainStep, coeffs, accum, count4, SamplesToDo); break;
    }
}
