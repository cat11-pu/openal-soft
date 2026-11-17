#define AL_ALEXT_PROTOTYPES

#include <gtest/gtest.h>

#include <AL/al.h>
#include <AL/alc.h>
#include <AL/alext.h>

#include <array>
#include <cstdlib>
#include <limits>
#include <thread>

namespace {

void ConfigureNullBackend()
{
#ifdef _WIN32
    _putenv_s("ALSOFT_DRIVERS", "null");
    _putenv_s("ALSOFT_LOGLEVEL", "0");
#else
    setenv("ALSOFT_DRIVERS", "null", 1);
    setenv("ALSOFT_LOGLEVEL", "0", 1);
#endif
}

struct NullBackendConfig {
    NullBackendConfig() { ConfigureNullBackend(); }
};
[[maybe_unused]] const NullBackendConfig gNullBackendConfig;

class AudioStabilityTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        ConfigureNullBackend();
    }

    void SetUp() override
    {
        mDevice = alcOpenDevice(nullptr);
        ASSERT_NE(mDevice, nullptr);
        mContext = alcCreateContext(mDevice, nullptr);
        ASSERT_NE(mContext, nullptr);
        ASSERT_EQ(alcMakeContextCurrent(mContext), ALC_TRUE);
        clearErrors();
    }

    void TearDown() override
    {
        if(mContext)
        {
            alcMakeContextCurrent(nullptr);
            alcDestroyContext(mContext);
        }
        if(mDevice)
            alcCloseDevice(mDevice);
    }

    void clearErrors()
    {
        while(alcGetError(mDevice) != ALC_NO_ERROR) { }
        while(alGetError() != AL_NO_ERROR) { }
    }

    ALCdevice *mDevice{};
    ALCcontext *mContext{};
};

TEST_F(AudioStabilityTest, RejectsInvalidDeviceAttributes)
{
    const auto badfreq = std::array{ALC_FREQUENCY, 0, 0};
    const auto badsources = std::array{ALC_MONO_SOURCES, -1, 0};
    const auto badsends = std::array{ALC_MAX_AUXILIARY_SENDS, -1, 0};
    const auto badhrtf = std::array{ALC_HRTF_SOFT, 12345, 0};
    const auto badattr = std::array{0x7fff, 1, 0};

    for(auto const *attrs : {badfreq.data(), badsources.data(), badsends.data(), badhrtf.data(),
        badattr.data()})
    {
        EXPECT_EQ(alcCreateContext(mDevice, attrs), nullptr);
        EXPECT_EQ(alcGetError(mDevice), ALC_INVALID_VALUE);
    }
    EXPECT_EQ(alcGetCurrentContext(), mContext);
}

TEST_F(AudioStabilityTest, RejectsNullArrays)
{
    alGenSources(1, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alDeleteSources(1, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alSourcePlayv(1, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alSourcePausev(1, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alSourceStopv(1, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alSourceRewindv(1, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alSourceQueueBuffers(1, 1, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alSourceUnqueueBuffers(1, 1, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alGenBuffers(1, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alDeleteBuffers(1, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alGenFilters(1, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alDeleteFilters(1, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alGenEffects(1, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alDeleteEffects(1, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alGenAuxiliaryEffectSlots(1, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alDeleteAuxiliaryEffectSlots(1, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);

    alFilteriv(1, AL_FILTER_TYPE, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alGetFilteri(1, AL_FILTER_TYPE, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alEffectiv(1, AL_EFFECT_TYPE, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alGetEffecti(1, AL_EFFECT_TYPE, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alAuxiliaryEffectSlotiv(1, AL_EFFECTSLOT_EFFECT, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    alGetAuxiliaryEffectSloti(1, AL_EFFECTSLOT_EFFECT, nullptr);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
}

TEST_F(AudioStabilityTest, HandlesStaleAndOutOfRangeSources)
{
    auto source = ALuint{};
    alGenSources(1, &source);
    ASSERT_EQ(alGetError(), AL_NO_ERROR);
    ASSERT_NE(source, 0u);

    alDeleteSources(1, &source);
    EXPECT_EQ(alGetError(), AL_NO_ERROR);
    alDeleteSources(1, &source);
    EXPECT_EQ(alGetError(), AL_INVALID_NAME);
    alSourcePlay(source);
    EXPECT_EQ(alGetError(), AL_INVALID_NAME);

    for(auto const id : std::array{0u, 1u, 63u, 64u, 1u<<20, 0x7fffffffu, 0xffffffffu})
    {
        alSourcePlay(id);
        EXPECT_EQ(alGetError(), AL_INVALID_NAME);
        alDeleteSources(1, &id);
        EXPECT_EQ(alGetError(), AL_INVALID_NAME);
    }
}

TEST_F(AudioStabilityTest, RejectsDuplicateDeleteAtomically)
{
    auto sources = std::array<ALuint, 2>{};
    alGenSources(2, sources.data());
    ASSERT_EQ(alGetError(), AL_NO_ERROR);
    ASSERT_NE(sources[0], sources[1]);

    const auto duplicates = std::array{sources[0], sources[0]};
    alDeleteSources(2, duplicates.data());
    EXPECT_EQ(alGetError(), AL_INVALID_NAME);
    EXPECT_EQ(alIsSource(sources[0]), AL_TRUE);
    EXPECT_EQ(alIsSource(sources[1]), AL_TRUE);

    alDeleteSources(2, sources.data());
    EXPECT_EQ(alGetError(), AL_NO_ERROR);
    EXPECT_EQ(alIsSource(sources[0]), AL_FALSE);
    EXPECT_EQ(alIsSource(sources[1]), AL_FALSE);
}

TEST_F(AudioStabilityTest, RepeatedInvalidTrafficIsStable)
{
    for(auto i = 0; i < 2000; ++i)
    {
        auto bad = static_cast<ALuint>((i & 1) ? 0xffffffffu : (1u<<20)+static_cast<ALuint>(i));
        alSourcePlay(bad);
        EXPECT_EQ(alGetError(), AL_INVALID_NAME);
        alDeleteSources(1, &bad);
        EXPECT_EQ(alGetError(), AL_INVALID_NAME);
        alSourcePlayv(0, nullptr);
        EXPECT_EQ(alGetError(), AL_NO_ERROR);
        alGenSources(1, nullptr);
        EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
    }
}

TEST_F(AudioStabilityTest, SurvivesReopenAndRejectsBadReopenAttributes)
{
    auto reopen = reinterpret_cast<LPALCREOPENDEVICESOFT>(
        alcGetProcAddress(nullptr, "alcReopenDeviceSOFT"));
    ASSERT_NE(reopen, nullptr);

    auto source = ALuint{};
    alGenSources(1, &source);
    ASSERT_EQ(alGetError(), AL_NO_ERROR);

    const auto badattrs = std::array{ALC_FREQUENCY, std::numeric_limits<ALCint>::min(), 0};
    EXPECT_EQ(alcResetDeviceSOFT(mDevice, badattrs.data()), ALC_FALSE);
    EXPECT_EQ(alcGetError(mDevice), ALC_INVALID_VALUE);
    EXPECT_EQ(alcGetCurrentContext(), mContext);
    EXPECT_EQ(alIsSource(source), AL_TRUE);

    EXPECT_EQ(reopen(mDevice, nullptr, badattrs.data()), ALC_FALSE);
    EXPECT_EQ(alcGetError(mDevice), ALC_INVALID_VALUE);
    EXPECT_EQ(alcGetCurrentContext(), mContext);
    EXPECT_EQ(alIsSource(source), AL_TRUE);

    const auto goodattrs = std::array{ALC_FREQUENCY, 48000, ALC_MONO_SOURCES, 256,
        ALC_STEREO_SOURCES, 1, ALC_MAX_AUXILIARY_SENDS, 2, 0};
    EXPECT_EQ(alcReopenDeviceSOFT(mDevice, nullptr, goodattrs.data()), ALC_TRUE);
    EXPECT_EQ(alcGetError(mDevice), ALC_NO_ERROR);
    EXPECT_EQ(alcGetCurrentContext(), mContext);
    EXPECT_EQ(alIsSource(source), AL_TRUE);

    alSourcePlay(source);
    EXPECT_EQ(alGetError(), AL_NO_ERROR);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    alSourceStop(source);
    EXPECT_EQ(alGetError(), AL_NO_ERROR);
    alDeleteSources(1, &source);
    EXPECT_EQ(alGetError(), AL_NO_ERROR);
}

TEST(AudioLifecycleTest, DoubleDestroyAndCloseAreExplicit)
{
    auto *device = alcOpenDevice(nullptr);
    ASSERT_NE(device, nullptr);
    auto *context = alcCreateContext(device, nullptr);
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(alcMakeContextCurrent(context), ALC_TRUE);
    while(alGetError() != AL_NO_ERROR) { }
    while(alcGetError(device) != ALC_NO_ERROR) { }

    alcMakeContextCurrent(nullptr);
    alcDestroyContext(context);
    alcDestroyContext(context);
    EXPECT_EQ(alcGetError(nullptr), ALC_INVALID_CONTEXT);

    EXPECT_EQ(alcCloseDevice(device), ALC_TRUE);
    EXPECT_EQ(alcCloseDevice(device), ALC_FALSE);
    EXPECT_EQ(alcGetError(nullptr), ALC_INVALID_DEVICE);
}

TEST(AudioLifecycleTest, DirectContextFunctionsRejectInvalidHandles)
{
    using SourcePlayDirect = void (AL_APIENTRY*)(ALCcontext*, ALuint);
    using GetErrorDirect = ALenum (AL_APIENTRY*)(ALCcontext*);

    auto source_play = reinterpret_cast<SourcePlayDirect>(
        alGetProcAddress("alSourcePlayDirect"));
    auto get_error = reinterpret_cast<GetErrorDirect>(alGetProcAddress("alGetErrorDirect"));
    ASSERT_NE(source_play, nullptr);
    ASSERT_NE(get_error, nullptr);

    source_play(nullptr, 1);
    EXPECT_EQ(alcGetError(nullptr), ALC_INVALID_CONTEXT);
    EXPECT_EQ(get_error(nullptr), AL_INVALID_OPERATION);
    EXPECT_EQ(alcGetError(nullptr), ALC_INVALID_CONTEXT);

    source_play(reinterpret_cast<ALCcontext*>(1), 1);
    EXPECT_EQ(alcGetError(nullptr), ALC_INVALID_CONTEXT);
}

} // namespace
