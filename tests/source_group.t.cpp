#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <AL/al.h>
#include <AL/alc.h>
#include <AL/efx.h>

#include "source_group.h"


namespace {

constexpr ALuint InvalidSourceId{0xdeadbe00u};
constexpr ALuint InvalidFilterId{0xdeadbe01u};

/* Counts live AL source/filter objects through RAII holders, so long-running
 * loops can prove they create nothing that outlives one iteration.
 */
struct LiveObjectCounters {
    int sources{0};
    int filters{0};
};

static LiveObjectCounters gLiveObjects;

class CountedSources {
public:
    explicit CountedSources(ALsizei count) : mIds(static_cast<std::size_t>(count), 0u)
    {
        if(count > 0)
        {
            alGenSources(count, mIds.data());
            gLiveObjects.sources += count;
        }
    }
    CountedSources(const CountedSources&) = delete;
    auto operator=(const CountedSources&) -> CountedSources& = delete;
    ~CountedSources()
    {
        if(!mIds.empty())
        {
            alDeleteSources(static_cast<ALsizei>(mIds.size()), mIds.data());
            gLiveObjects.sources -= static_cast<ALsizei>(mIds.size());
        }
    }

    [[nodiscard]] auto get() const noexcept -> std::span<const ALuint> { return mIds; }
    [[nodiscard]] auto size() const noexcept -> std::size_t { return mIds.size(); }
    [[nodiscard]] auto operator[](std::size_t index) const noexcept -> ALuint { return mIds[index]; }

private:
    std::vector<ALuint> mIds;
};

class CountedFilter {
public:
    CountedFilter()
    {
        alGenFilters(1, &mId);
        if(mId != 0)
        {
            alFilteri(mId, AL_FILTER_TYPE, AL_FILTER_LOWPASS);
            ++gLiveObjects.filters;
        }
    }
    CountedFilter(const CountedFilter&) = delete;
    auto operator=(const CountedFilter&) -> CountedFilter& = delete;
    ~CountedFilter()
    {
        if(mId != 0)
        {
            alDeleteFilters(1, &mId);
            --gLiveObjects.filters;
        }
    }

    [[nodiscard]] auto id() const noexcept -> ALuint { return mId; }

private:
    ALuint mId{0};
};

auto getGain(ALuint source) -> ALfloat
{
    ALfloat value{0.0f};
    alGetSourcef(source, AL_GAIN, &value);
    return value;
}

auto getDistanceModel(ALuint source) -> ALint
{
    ALint value{0};
    alGetSourcei(source, AL_DISTANCE_MODEL, &value);
    return value;
}

auto floatsEqual(ALfloat lhs, ALfloat rhs) -> bool
{
    if(std::isnan(lhs)) return std::isnan(rhs);
    return std::abs(lhs - rhs) <= 1.0e-6f * std::max({1.0f, std::abs(lhs), std::abs(rhs)});
}

constexpr std::array<ALenum, 7> kDistanceModels{
    AL_NONE,
    AL_INVERSE_DISTANCE,
    AL_INVERSE_DISTANCE_CLAMPED,
    AL_LINEAR_DISTANCE,
    AL_LINEAR_DISTANCE_CLAMPED,
    AL_EXPONENT_DISTANCE,
    AL_EXPONENT_DISTANCE_CLAMPED,
};

} // namespace


class SourceGroupTest : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        mDevice = alcOpenDevice(nullptr);
        ASSERT_NE(mDevice, nullptr);
        mEfxSupported = alcIsExtensionPresent(mDevice, "ALC_EXT_EFX") != ALC_FALSE;
        const ALCint attrs[]{ALC_MAX_AUXILIARY_SENDS, mEfxSupported ? 1 : 0, 0};
        mContext = alcCreateContext(mDevice, mEfxSupported ? attrs : nullptr);
        ASSERT_NE(mContext, nullptr);
        ASSERT_TRUE(alcMakeContextCurrent(mContext) != 0);
        alGetError();
    }

    static void TearDownTestSuite()
    {
        if(mContext)
        {
            alcMakeContextCurrent(nullptr);
            alcDestroyContext(mContext);
        }
        if(mDevice) alcCloseDevice(mDevice);
        mContext = nullptr;
        mDevice = nullptr;
    }

    void SetUp() override { EXPECT_EQ(alGetError(), AL_NO_ERROR); }
    void TearDown() override { EXPECT_EQ(alGetError(), AL_NO_ERROR); }

    static ALCdevice* mDevice;
    static ALCcontext* mContext;
    static bool mEfxSupported;
};

ALCdevice* SourceGroupTest::mDevice{nullptr};
ALCcontext* SourceGroupTest::mContext{nullptr};
bool SourceGroupTest::mEfxSupported{false};


TEST_F(SourceGroupTest, EmptyGroupIsNoop)
{
    altest::SourceGroup group;
    EXPECT_TRUE(group.empty());
    EXPECT_EQ(group.size(), 0u);
    EXPECT_EQ(group.setGain(0.5f), 0u);
    EXPECT_EQ(group.setDistanceModel(AL_LINEAR_DISTANCE), 0u);
    EXPECT_EQ(group.setGainFilterDistance(0.5f, 0u, AL_LINEAR_DISTANCE), 0u);
    EXPECT_EQ(alGetError(), AL_NO_ERROR);
}

TEST_F(SourceGroupTest, AddRemoveAndDedup)
{
    CountedSources sources{4};
    ASSERT_EQ(alGetError(), AL_NO_ERROR);

    altest::SourceGroup group;
    group.addSource(sources[0]);
    group.addSource(sources[0]);
    group.addSources(sources.get().subspan(1, 3));
    ASSERT_EQ(group.size(), 4u);

    EXPECT_TRUE(group.contains(sources[2]));
    EXPECT_TRUE(group.removeSource(sources[2]));
    EXPECT_FALSE(group.contains(sources[2]));
    EXPECT_FALSE(group.removeSource(sources[2]));
    EXPECT_EQ(group.size(), 3u);

    group.clear();
    EXPECT_TRUE(group.empty());

    /* Clearing the group does not delete the AL sources. */
    EXPECT_EQ(alIsSource(sources[2]), AL_TRUE);
}

TEST_F(SourceGroupTest, BatchGainAppliesToEveryMember)
{
    CountedSources sources{5};
    altest::SourceGroup group{sources.get()};

    EXPECT_EQ(group.setGain(0.42f), 5u);
    for(ALuint source : sources.get())
        EXPECT_PRED2(floatsEqual, getGain(source), 0.42f);
}

TEST_F(SourceGroupTest, BatchDistanceModelCoversAllEnums)
{
    CountedSources sources{kDistanceModels.size()};
    altest::SourceGroup group{sources.get()};

    for(std::size_t i{0}; i < kDistanceModels.size(); ++i)
    {
        ASSERT_EQ(group.setDistanceModel(kDistanceModels[i]), group.size());
        EXPECT_EQ(getDistanceModel(sources[i]), static_cast<ALint>(kDistanceModels[i]));
    }
}

TEST_F(SourceGroupTest, BatchDirectFilter)
{
    if(!mEfxSupported) GTEST_SKIP() << "ALC_EXT_EFX not supported";

    CountedSources sources{3};
    CountedFilter filter;
    ASSERT_NE(filter.id(), 0u);
    altest::SourceGroup group{sources.get()};

    /* AL_DIRECT_FILTER is write-only in OpenAL Soft (a get raises
     * AL_INVALID_ENUM), so a clean error state after each batch is the
     * observable contract, plus rejection of unknown filter IDs.
     */
    EXPECT_EQ(group.setDirectFilter(filter.id()), 3u);
    ASSERT_EQ(alGetError(), AL_NO_ERROR);

    EXPECT_EQ(group.setDirectFilter(0u), 3u);
    EXPECT_EQ(alGetError(), AL_NO_ERROR);

    EXPECT_EQ(group.setDirectFilter(InvalidFilterId), 3u);
    EXPECT_EQ(alGetError(), AL_INVALID_NAME);
}

TEST_F(SourceGroupTest, InvalidGainLeavesErrorAndKeepsFirstError)
{
    CountedSources sources{3};
    altest::SourceGroup group{sources.get()};

    EXPECT_EQ(group.setGain(-1.0f), 3u);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);

    /* First error is retained even when more invalid calls follow. */
    alSourcef(sources[0], AL_GAIN, -2.0f);
    alSourcef(sources[1], AL_GAIN, -2.0f);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);
}

TEST_F(SourceGroupTest, InvalidDistanceModelRaisesErrorPerSource)
{
    CountedSources sources{3};
    altest::SourceGroup group{sources.get()};

    EXPECT_EQ(group.setDistanceModel(0x7777), 3u);
    EXPECT_EQ(alGetError(), AL_INVALID_VALUE);

    /* A failed distance-model update does not touch unrelated state. */
    EXPECT_EQ(getDistanceModel(sources[0]),
        static_cast<ALint>(AL_INVERSE_DISTANCE_CLAMPED));
}

TEST_F(SourceGroupTest, InvalidSourceRaisesInvalidName)
{
    CountedSources sources{2};
    altest::SourceGroup group;
    group.addSource(sources[0]);
    group.addSource(InvalidSourceId);
    group.addSource(sources[1]);

    EXPECT_FALSE(group.validateSources());
    EXPECT_EQ(group.setGain(0.7f), 3u);
    EXPECT_EQ(alGetError(), AL_INVALID_NAME);
}

TEST_F(SourceGroupTest, InvalidFilterRaisesInvalidName)
{
    if(!mEfxSupported) GTEST_SKIP() << "ALC_EXT_EFX not supported";

    CountedSources sources{2};
    altest::SourceGroup group{sources.get()};

    EXPECT_EQ(group.setDirectFilter(InvalidFilterId), 2u);
    EXPECT_EQ(alGetError(), AL_INVALID_NAME);
}

TEST_F(SourceGroupTest, BatchMatchesPerSourceCallsSampleBySample)
{
    constexpr ALsizei SourceCount{16};
    constexpr std::size_t RoundCount{24};

    CountedSources batchSources{SourceCount};
    CountedSources referenceSources{SourceCount};
    CountedFilter filterA;
    CountedFilter filterB;
    ASSERT_TRUE(mEfxSupported ? filterA.id() != 0u && filterB.id() != 0u : true);

    altest::SourceGroup batchGroup{batchSources.get()};
    altest::SourceGroup referenceGroup{referenceSources.get()};

    /* Filter IDs stay 0 on devices without EFX, so the direct-filter setter
     * becomes a no-op reset instead of an invalid-enum lookup.
     */
    const std::array<ALuint, 4> filters{mEfxSupported ? filterA.id() : 0u,
        mEfxSupported ? filterB.id() : 0u, 0u, 0u};

    for(std::size_t round{0}; round < RoundCount; ++round)
    {
        const ALfloat gain{0.1f + 0.03f * static_cast<ALfloat>(round)};
        const ALuint filter{filters[round % filters.size()]};
        const ALenum model{kDistanceModels[round % kDistanceModels.size()]};

        /* Batch path. */
        EXPECT_EQ(batchGroup.setGainFilterDistance(gain, filter, model),
            static_cast<std::size_t>(SourceCount));
        EXPECT_EQ(alGetError(), AL_NO_ERROR);

        /* Reference path: the same three setters, one source at a time. */
        for(ALuint source : referenceSources.get())
        {
            alSourcef(source, AL_GAIN, gain);
            if(mEfxSupported || filter == 0u)
                alSourcei(source, AL_DIRECT_FILTER, static_cast<ALint>(filter));
            alSourcei(source, AL_DISTANCE_MODEL, static_cast<ALint>(model));
        }
        EXPECT_EQ(alGetError(), AL_NO_ERROR);

        for(ALsizei i{0}; i < SourceCount; ++i)
        {
            SCOPED_TRACE(::testing::Message() << "round=" << round << " source=" << i);
            EXPECT_PRED2(floatsEqual, getGain(batchSources[i]), getGain(referenceSources[i]));
            EXPECT_EQ(getDistanceModel(batchSources[i]), getDistanceModel(referenceSources[i]));
        }
    }
}

TEST_F(SourceGroupTest, IndependentSourcesCanStillBeUpdatedIndividually)
{
    CountedSources sources{4};
    altest::SourceGroup group{sources.get()};

    ASSERT_EQ(group.setGain(0.3f), 4u);
    ASSERT_EQ(alGetError(), AL_NO_ERROR);

    /* A per-source override after the batch only changes that source. */
    alSourcef(sources[2], AL_GAIN, 0.9f);
    ASSERT_EQ(alGetError(), AL_NO_ERROR);

    for(ALsizei i{0}; i < 4; ++i)
        EXPECT_PRED2(floatsEqual, getGain(sources[i]), i == 2 ? 0.9f : 0.3f);
}

TEST_F(SourceGroupTest, TenThousandContinuousUpdatesDoNotLeak)
{
    constexpr ALsizei SourceCount{8};
    constexpr int IterationCount{10000};

    CountedSources sources{SourceCount};
    CountedFilter filter;
    if(mEfxSupported)
    {
        ASSERT_NE(filter.id(), 0u);
    }
    altest::SourceGroup group{sources.get()};

    const LiveObjectCounters beforeLoop{gLiveObjects};
    ASSERT_EQ(alGetError(), AL_NO_ERROR);

    for(int iteration{0}; iteration < IterationCount; ++iteration)
    {
        const ALfloat gain{0.5f + 0.5f
            * static_cast<ALfloat>(iteration % 100) / 100.0f};
        const ALuint filterId{(mEfxSupported && iteration % 2 == 0) ? filter.id() : 0u};
        const ALenum model{kDistanceModels[iteration % kDistanceModels.size()]};

        const auto processed = group.setGainFilterDistance(gain, filterId, model);
        ASSERT_EQ(processed, static_cast<std::size_t>(SourceCount));
        ASSERT_EQ(alGetError(), AL_NO_ERROR);

        /* A batch update must never allocate AL objects. */
        EXPECT_EQ(gLiveObjects.sources, beforeLoop.sources);
        EXPECT_EQ(gLiveObjects.filters, beforeLoop.filters);
    }

    /* Final state of every member reflects the last iteration. */
    const ALfloat expectedGain{0.5f + 0.5f * 99.0f / 100.0f};
    const ALenum expectedModel{kDistanceModels[(IterationCount - 1) % kDistanceModels.size()]};
    for(ALuint source : sources.get())
    {
        EXPECT_PRED2(floatsEqual, getGain(source), expectedGain);
        EXPECT_EQ(getDistanceModel(source), static_cast<ALint>(expectedModel));
    }

    /* Live object counts are unchanged after 10k iterations, no error pending. */
    EXPECT_EQ(gLiveObjects.sources, beforeLoop.sources);
    EXPECT_EQ(gLiveObjects.filters, beforeLoop.filters);
    EXPECT_EQ(alGetError(), AL_NO_ERROR);
}
