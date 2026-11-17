#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <vector>

#include <AL/al.h>
#include <AL/alc.h>
#include <AL/efx.h>

#include "source_group.h"

namespace {

constexpr ALsizei SourceCount{16};
constexpr int IterationCount{10000};
constexpr ALenum DistanceModels[]{
    AL_NONE, AL_INVERSE_DISTANCE, AL_INVERSE_DISTANCE_CLAMPED,
    AL_LINEAR_DISTANCE, AL_LINEAR_DISTANCE_CLAMPED,
    AL_EXPONENT_DISTANCE, AL_EXPONENT_DISTANCE_CLAMPED,
};

auto gainOf(ALuint source) -> ALfloat
{
    ALfloat value{0.0f};
    alGetSourcef(source, AL_GAIN, &value);
    return value;
}

auto modelOf(ALuint source) -> ALint
{
    ALint value{0};
    alGetSourcei(source, AL_DISTANCE_MODEL, &value);
    return value;
}

} // namespace

int main()
{
    ALCdevice *device{alcOpenDevice(nullptr)};
    if(!device)
    {
        std::fprintf(stderr, "alcOpenDevice failed\n");
        return EXIT_FAILURE;
    }
    const bool efx{alcIsExtensionPresent(device, "ALC_EXT_EFX") != AL_FALSE};
    const ALCint attrs[]{ALC_MAX_AUXILIARY_SENDS, efx ? 1 : 0, 0};
    ALCcontext *context{alcCreateContext(device, efx ? attrs : nullptr)};
    if(!context || alcMakeContextCurrent(context) == 0)
    {
        std::fprintf(stderr, "context setup failed\n");
        return EXIT_FAILURE;
    }

    std::vector<ALuint> batchIds(SourceCount, 0u);
    std::vector<ALuint> refIds(SourceCount, 0u);
    alGenSources(SourceCount, batchIds.data());
    alGenSources(SourceCount, refIds.data());
    if(alGetError() != AL_NO_ERROR)
    {
        std::fprintf(stderr, "alGenSources failed\n");
        return EXIT_FAILURE;
    }

    ALuint filter{0};
    if(efx)
    {
        alGenFilters(1, &filter);
        alFilteri(filter, AL_FILTER_TYPE, AL_FILTER_LOWPASS);
        alFilterf(filter, AL_LOWPASS_GAIN, 0.7f);
        alFilterf(filter, AL_LOWPASS_GAINHF, 0.4f);
    }

    altest::SourceGroup batch{batchIds};

    /* ---- consistency: 24 rounds, batch path vs per-source reference ---- */
    int mismatches{0};
    constexpr int Rounds{24};
    for(int round{0}; round < Rounds; ++round)
    {
        const ALfloat gain{0.1f + 0.03f * static_cast<ALfloat>(round)};
        const ALuint filterId{efx && round % 2 == 0 ? filter : 0u};
        const ALenum model{DistanceModels[round % 7]};

        batch.setGainFilterDistance(gain, filterId, model);
        if(alGetError() != AL_NO_ERROR) ++mismatches;

        for(ALuint source : refIds)
        {
            alSourcef(source, AL_GAIN, gain);
            if(efx || filterId == 0u)
                alSourcei(source, AL_DIRECT_FILTER, static_cast<ALint>(filterId));
            alSourcei(source, AL_DISTANCE_MODEL, static_cast<ALint>(model));
        }
        if(alGetError() != AL_NO_ERROR) ++mismatches;

        for(ALsizei i{0}; i < SourceCount; ++i)
        {
            if(gainOf(batchIds[i]) != gainOf(refIds[i])) ++mismatches;
            if(modelOf(batchIds[i]) != modelOf(refIds[i])) ++mismatches;
        }
    }

    std::printf("consistency rounds      : %d over %d sources (%d sample pairs)\n",
        Rounds, SourceCount, Rounds * SourceCount * 2);
    std::printf("consistency mismatches  : %d\n", mismatches);
    std::printf("sample[0] batch gain    : %.7f  ref gain: %.7f\n",
        gainOf(batchIds[0]), gainOf(refIds[0]));
    std::printf("sample[0] batch model   : %d  ref model: %d\n",
        modelOf(batchIds[0]), modelOf(refIds[0]));

    /* ---- 10k continuous updates: timing + object-leak guard ---- */
    const auto start{std::chrono::steady_clock::now()};
    for(int iteration{0}; iteration < IterationCount; ++iteration)
    {
        const ALfloat gain{0.5f + 0.5f
            * static_cast<ALfloat>(iteration % 100) / 100.0f};
        const ALuint filterId{efx && iteration % 2 == 0 ? filter : 0u};
        const ALenum model{DistanceModels[iteration % 7]};
        batch.setGainFilterDistance(gain, filterId, model);
    }
    const auto end{std::chrono::steady_clock::now()};
    const ALenum pendingError{alGetError()};

    const auto us{std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()};
    const auto calls{IterationCount * SourceCount * 3};
    std::printf("10k iterations          : %d updates x %d sources x 3 setters = %d AL calls\n",
        IterationCount, SourceCount, calls);
    std::printf("elapsed                 : %lld us (%.3f us/iteration, %.1f ns/call)\n",
        static_cast<long long>(us), static_cast<double>(us) / IterationCount,
        static_cast<double>(us) * 1000.0 / calls);
    std::printf("pending AL error        : 0x%x (%s)\n", pendingError,
        pendingError == AL_NO_ERROR ? "none" : "UNEXPECTED");

    /* The batch path never allocates; only the two source arrays and the
     * optional filter exist, and all are deleted below.
     */
    const std::ptrdiff_t expectedLiveSources{SourceCount * 2};
    std::printf("live sources at end     : %td (expected %td)\n",
        static_cast<std::ptrdiff_t>(batchIds.size() + refIds.size()), expectedLiveSources);

    if(filter) alDeleteFilters(1, &filter);
    alDeleteSources(SourceCount, batchIds.data());
    alDeleteSources(SourceCount, refIds.data());
    alcMakeContextCurrent(nullptr);
    alcDestroyContext(context);
    alcCloseDevice(device);
    return mismatches == 0 && pendingError == AL_NO_ERROR ? EXIT_SUCCESS : EXIT_FAILURE;
}
